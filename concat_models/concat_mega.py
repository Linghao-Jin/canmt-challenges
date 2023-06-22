from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor

from fairseq.modules import LayerDropModuleList, MegaEncoderLayer
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.mega import (
    MegaModel,
    MegaEncoder,
    MegaDecoder,
)
from fairseq.models.mega import (
    base_architecture as mega_base_architecture,
    mega_wmt_en_de,
    transformer_vaswani_wmt_en_de_big,
)

@register_model("concat_mega")
class ConcatMegaModel(MegaModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        MegaModel.add_args(parser)
        
        parser.add_argument(
            "--coword-dropout",
            default=0.0,
            type=float,
            help="if set to value>0, randomly drops source tokens",
        )
        parser.add_argument(
            "--coword-dropout-type",
            choices=("sample", "predefined_sample", "whole", "suffix"),
            default=None,
            help="type of coword dropout to use. NOTE: only sample is used"
            "used in the paper",
        )
        # parser.add_argument('--source-context-size', type=int, default=1)
        # parser.add_argument('--target-context-size', type=int, default=1)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return ConcatMegaEncoder(
            args,
            src_dict,
            embed_tokens,
            coword_dropout_prob=getattr(args, "coword_dropout", 0.0),
            coword_dropout_type=getattr(args, "coword_dropout_type", "sample"),
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return ConcatMegaDecoder(
            args,
            tgt_dict,
            embed_tokens,
        )


    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        src_ctx_tokens=None,
        src_ctx_lengths=None,
        tgt_ctx_tokens=None,
        tgt_ctx_lengths=None,
        src_sample_probs=None,
        prev_output_tokens=None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None
    ): 

        encoder_out = self.encoder(
            src_tokens=src_tokens,
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
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )   

        return decoder_out


class ConcatMegaEncoder(MegaEncoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        coword_dropout_type="sample",
        coword_dropout_prob=0.0,
    ):
        super().__init__(args, dictionary, embed_tokens)
        self.coword_dropout_type = coword_dropout_type
        self.coword_dropout_prob = coword_dropout_prob
        # TODO: add this a variable token
        self.mask_id = dictionary.index("<mask>")
        self.args = args
        self.num_layers = len(self.layers)




    def forward(
        self,
        src_tokens,
        src_lengths,
        src_ctx_tokens,
        src_ctx_lengths,
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

        ## compute num paddings
        if self.args.source_context_size > 0:
            input_tokens = torch.cat([src_ctx_tokens, src_tokens], axis=1)
        else:
            input_tokens = src_tokens
        
        # print(f'input toks: {input_tokens}')
        # print(f'embed scale: {self.embed_scale}')
        # print(f'embed tokens: {self.embed_tokens}')
        
        x = encoder_embedding =  self.embed_scale * self.embed_tokens(input_tokens) # forward_embedding - TODO might
        # print(f'embed: {x}')
        if self.embed_norm is not None:
            x = self.embed_norm(x)
        x = self.embedding_dropout(x)



        # account for padding while computing the representation
        padding_mask = input_tokens.eq(self.padding_idx)

        if not padding_mask.any():
            padding_mask = None



        if padding_mask is not None:
            # B x T aka 4 x 41
            inverse_mask = 1.0 - padding_mask.type_as(x) 
            x =x * inverse_mask.unsqueeze(-1)
        else:
            inverse_mask = None   
          
        
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)  


        x_encoder_states = [] if return_all_hiddens else None
        for layer in self.layers:
            x = layer(x, padding_mask)
            if return_all_hiddens:
                assert x_encoder_states is not None
                x_encoder_states.append(x)

        
        if self.final_norm is not None:
            x = self.final_norm(x)

        if inverse_mask is not None:
            x = x * inverse_mask.transpose(0, 1).unsqueeze(-1)


        return EncoderOut(
            encoder_out=x,
            encoder_padding_mask=padding_mask,
            encoder_embedding=encoder_embedding,
            encoder_states=x_encoder_states,
            src_tokens=None,
            src_lengths=None,
        )


class ConcatMegaDecoder(MegaDecoder):
    def __init__(
        self, args, dictionary, embed_tokens
    ):
        super().__init__(args, dictionary, embed_tokens) 


    def forward(
        self,
        prev_output_tokens,
        context_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        #alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            context_tokens (LongTensor): context tokens (ie a prefix
                to prev_output_tokens), shape `(batch, tgt_ctx_len)`
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            context_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
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
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            context_tokens, # new
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
        )


    def extract_features_scriptable(
        self,
        prev_output_tokens,
        context_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer =  self.num_layers - 1

        if self.args.target_context_size > 0:
            input_tokens = torch.cat([context_tokens, prev_output_tokens], axis=1)
            context_end_id = context_tokens.size(1) # where targ context stops
        else:
            input_tokens = prev_output_tokens  

        # if self.embed_positions is not None:
        #     positions = self.embed_positions(
        #         input_tokens, incremental_state=incremental_state
        #     )   
        # else:
        #     positions = None

        if incremental_state is not None:
            input_tokens = input_tokens[:, -1:]
            context_end_id = 0
            # if positions is not None:
            #     positions = positions[:, -1:]
       
        bsz, seq_len = input_tokens.size()
       

        x = self.embed_scale * self.embed_tokens(input_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if self.embed_norm is not None:
            x = self.embed_norm(x)

        x = self.embedding_dropout(x)

        decoder_padding_mask = input_tokens.eq(self.padding_idx)
        
        if not decoder_padding_mask.any():
            decoder_padding_mask = None

        # account for padding while computing the representation
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
                incremental_state is None 
            ) and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
            x, layer_attn, _ = layer(
                x=x, # input to the layer 
                encoder_out=encoder_out.encoder_out,  
                encoder_padding_mask=encoder_out.encoder_padding_mask, # encodier padding mask
                incremental_state=incremental_state,
                attn_mask = self_attn_mask, 
                need_attn=bool((idx == alignment_layer)),
                decoder_padding_mask = decoder_padding_mask,
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)   

        # change: final norm after removing context
        if self.final_norm is not None:
            x = self.final_norm(x)
    
        if inverse_mask is not None:
            x = x * inverse_mask.transpose(0, 1).unsqueeze(-1)

        # remove context
        # context + targ is decoded, but we only want to generate targ
        if self.args.target_context_size > 0:
            x = x[context_end_id:]


        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": [attn], "inner_states": inner_states}

@register_model_architecture("concat_mega", "concat_mega")
def concat_mega_base_architecture(args):
    mega_base_architecture(args) # from mega

# @register_model_architecture("concat_transformer", "concat_transformer")
# def concat_transformer_base_architecture(args):
    # transformer_base_architecture(args)
# 
# 
@register_model_architecture("concat_mega", "concat_mega_iwslt")
def concat_mega_iwslt_architecture(args):
    mega_wmt_en_de(args)
# 
# 
@register_model_architecture("concat_mega", "concat_mega_big")
def concat_mega_big_architecture(args):
    transformer_vaswani_wmt_en_de_big(args)