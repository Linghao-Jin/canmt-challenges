import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import LayerDropModuleList
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    SequenceNorm,
    MegaEncoderLayer,
    MegaDecoderLayer,
)
from torch import Tensor
from fairseq.models.mega import (
    MegaModel,
    MegaEncoder,
    MegaDecoder,
    base_architecture,
    transformer_vaswani_wmt_en_de_big,
)
from fairseq.models.transformer import (
    transformer_iwslt_de_en,
)


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("contextual_mega")
class ContextualMegaModel(MegaModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        MegaModel.add_args(parser)
        parser.add_argument(
            "--context-loss",
            default=False,
            action="store_true",
            help="if set, trains to predict target context tokens",
        )
        parser.add_argument(
            "--coword-dropout",
            default=0.0,
            type=float,
            help="if set to value>0, randomly drops source tokens",
        )
        parser.add_argument(
            "--coword-dropout-type",
            choices=("sample", "predefined_sample", "whole", "suffix"),
            default="sample",
            help="type of coword dropout to use. NOTE: only sample is used"
            "used in the paper",
        )
        parser.add_argument(
            "--multi-encoder",
            default=False,
            action="store_true",
            help="whether to use multi-encoder in the source side",
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return ContextualMegaEncoder(
            args,
            src_dict,
            embed_tokens,
            src_ctx_tokens=None,
            src_ctx_lengths=None,
            multi_encoder=getattr(args, "multi_encoder", False),
            coword_dropout_prob=getattr(args, "coword_dropout", 0.0),
            coword_dropout_type=getattr(args, "coword_dropout_type", "sample"),
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return ContextualMegaDecoder(
            args,
            tgt_dict,
            embed_tokens,
            multi_encoder=getattr(args, "multi_encoder", False),
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        src_ctx_tokens,
        src_ctx_lengths,
        tgt_ctx_tokens=None,
        tgt_ctx_lengths=None,
        src_sample_probs=None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            src_ctx_tokens=src_ctx_tokens,
            src_ctx_lengths=src_ctx_lengths,
            src_sample_probs=src_sample_probs,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            context_tokens=tgt_ctx_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out


class ContextualMegaEncoder(MegaEncoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        src_ctx_tokens=None,
        src_ctx_lengths=None,
        multi_encoder=False,
        coword_dropout_type="sample",
        coword_dropout_prob=0.0,
    ):
        super().__init__(args, dictionary, embed_tokens)
        self.coword_dropout_type = coword_dropout_type
        self.coword_dropout_prob = coword_dropout_prob

        self.mask_id = dictionary.index("<mask>")
        self.multi_encoder = multi_encoder
        if self.multi_encoder:
            if self.encoder_layerdrop > 0.0:
                self.context_layers = LayerDropModuleList(p=self.encoder_layerdrop)
            else:
                self.context_layers = nn.ModuleList([])

            self.context_layers.extend(
                [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
            )

        self.num_layers = len(self.layers)

    def build_encoder_layer(self, args):
        return MegaEncoderLayer(args)

    def forward(
        self,
        src_tokens,
        src_lengths,
        src_ctx_tokens=None,
        src_ctx_lengths=None,
        tgt_tokens=None,
        tgt_lengths=None,
        tgt_ctx_tokens=None,
        tgt_ctx_lengths=None,
        src_sample_probs=None,
        return_all_hiddens: bool = False,
    ):
        # if source dropout enabled, randomly drop tokens from input
        if self.training and self.coword_dropout_type is not None:
            if self.coword_dropout_type == "sample":
                padding_mask = src_tokens.eq(self.padding_idx)
                mask_token = torch.tensor(self.mask_id).to(src_tokens)
                probs = torch.ones_like(src_tokens) * self.coword_dropout_prob
                mask = torch.logical_and(
                    torch.bernoulli(probs), torch.logical_not(padding_mask)
                )
                src_tokens = torch.where(mask == 0, src_tokens, mask_token)
            elif self.coword_dropout_type == "predefined_sample":
                # This is used for sampling with token specific probabilies
                # NOTE: this was not used in the paper
                assert (
                    src_sample_probs is not None
                ), "need sample probabilities as a given"
                padding_mask = src_tokens.eq(self.padding_idx)
                mask_token = torch.tensor(self.mask_id).to(src_tokens)
                mask = torch.logical_and(
                    torch.bernoulli(src_sample_probs), torch.logical_not(padding_mask)
                )
                src_tokens = torch.where(mask == 0, src_tokens, mask_token)
            elif self.coword_dropout_type == "whole":
                # make tensor with a single token (mask token)
                # NOTE: not used in the paper
                mask_samples = torch.zeros_like(src_tokens).to(src_tokens)
                mask_samples[mask_samples == 0] = self.padding_idx
                mask_samples[:, 0] = self.mask_id
                # replace samples by this tensor based on bernoulli
                probs = torch.ones((src_tokens.size(0),)) * self.coword_dropout_prob
                mask = torch.bernoulli(probs).to(src_tokens)
                mask = torch.unsqueeze(mask, -1).repeat(1, src_tokens.size(1))
                src_tokens = torch.where(mask == 0, src_tokens, mask_samples)
            else:
                raise ValueError(
                    f"unknown type of source dropout {self.coword_dropout_type}"
                )
        if src_ctx_tokens != None:
            input_tokens = torch.cat([src_ctx_tokens, src_tokens], axis=1)
        else:
            input_tokens = src_tokens
        x = encoder_embedding = self.embed_scale * self.embed_tokens(input_tokens)
        if self.embed_norm is not None:
            x = self.embed_norm(x)
        x = self.embedding_dropout(x)

        padding_mask = input_tokens.eq(self.padding_idx)
        ######
        ## my assumption here is that there cannot be padding toks
        # between the source and the context sentence
        if not padding_mask.any():
            padding_mask = None

        if padding_mask is not None:
            inverse_mask = 1.0 - padding_mask.type_as(x)
            x = x * inverse_mask.unsqueeze(-1)
        else:
            inverse_mask = None
        ######

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = [] if return_all_hiddens else None
        # encoder layers
        for layer in self.layers:
            x = layer(x, padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.final_norm is not None:
            x = self.final_norm(x)

        if inverse_mask is not None:
            x = x * inverse_mask.transpose(0, 1).unsqueeze(-1)

        return EncoderOut(
            encoder_out=x,
            encoder_padding_mask=padding_mask,
            encoder_embedding=encoder_embedding,
            encoder_states=encoder_states,
            src_tokens=None,
            src_lengths=None,
        )


class ContextualMegaDecoder(MegaDecoder):
    def __init__(
        self, args, dictionary, embed_tokens, multi_encoder=False, no_encoder_attn=False
    ):
        super().__init__(args, dictionary, embed_tokens)
        self.multi_encoder = multi_encoder
        if self.multi_encoder:
            if self.decoder_layerdrop > 0.0:
                self.context_layers = LayerDropModuleList(p=self.decoder_layerdrop)
            else:
                self.context_layers = nn.ModuleList([])

            self.context_layers.extend(
                [self.build_encoder_layer(args) for i in range(args.decoder_layers)]
            )

    def build_encoder_layer(self, args):
        return MegaEncoderLayer(args)

    def forward(
        self,
        prev_output_tokens,
        context_tokens=None,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        x, extra = self.extract_features(
            prev_output_tokens,
            context_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        context_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            context_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        context_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        if self.multi_encoder:
            ctx_padding_mask = context_tokens.eq(self.padding_idx)
            ctx_x, _ = self.forward_embedding(context_tokens)
            # B x T x C -> T x B x C
            ctx_x = ctx_x.transpose(0, 1)
            for layer in self.context_layers:
                ctx_x = layer(ctx_x, ctx_padding_mask)

            if self.layer_norm is not None:
                ctx_x = self.layer_norm(ctx_x)

            input_tokens = prev_output_tokens
        else:
            if context_tokens != None:
                input_tokens = torch.cat([context_tokens, prev_output_tokens], axis=1)
                context_end_id = context_tokens.size(1)
            else:
                input_tokens = prev_output_tokens
                context_end_id = 0

        # embed positions
        # if self.embed_positions is not None:
        #     # concat context_tokens to input
        #     # FIXME: this is really simple
        #     positions = self.embed_positions(
        #         input_tokens, incremental_state=incremental_state
        #     )
        # else:
        #     positions = None

        if incremental_state is not None and len(incremental_state) > 0:
            input_tokens = input_tokens[:, -1:]
            context_end_id = 0
            # if positions is not None:
            #     positions = positions[:, -1:]

        # embed tokens
        x = self.embed_scale * self.embed_tokens(input_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if self.embed_norm is not None:
            x = self.embed_norm(x)

        x = self.embedding_dropout(x)

        decoder_padding_mask = input_tokens.eq(self.padding_idx)
        if not decoder_padding_mask.any():
            decoder_padding_mask = None

        if decoder_padding_mask is not None:
            # B x T
            inverse_mask = 1.0 - decoder_padding_mask.type_as(x)
            x = x * inverse_mask.unsqueeze(-1)
        else:
            inverse_mask = None

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if (
                incremental_state is None or len(incremental_state) == 0
            ) and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out
                if encoder_out is not None
                else None,  # cross_attn
                encoder_out.encoder_padding_mask
                if encoder_out is not None
                else None,  # cross_attn_mask
                incremental_state,
                attn_mask=self_attn_mask,
                decoder_padding_mask=decoder_padding_mask,
                need_attn=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if self.final_norm is not None:
            x = self.final_norm(x)

        if inverse_mask is not None:
            x = x * inverse_mask.transpose(0, 1).unsqueeze(-1)

        # remove context
        if not self.multi_encoder:
            x = x[context_end_id:]

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": [attn], "inner_states": inner_states}


@register_model_architecture("contextual_mega", "contextual_mega")
def contextual_mega_base_architecture(args):
    base_architecture(args)


@register_model_architecture("contextual_mega", "contextual_mega_iwslt")
def contextual_mega_iwslt_architecture(args):
    transformer_iwslt_de_en(args)


@register_model_architecture("contextual_mega", "contextual_mega_big")
def contextual_mega_big_architecture(args):
    transformer_vaswani_wmt_en_de_big(args)
