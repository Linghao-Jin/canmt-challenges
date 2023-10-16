import argparse
from collections import Counter

import torch
import sacrebleu
from comet import download_model, load_from_checkpoint
from blonde import BLONDE


def make_n_sentence_corpus(
    full_corpus, docids, context_size, n_sentence, break_tag="<brk>"
):
    corpus = []
    prev_docid = -1
    docs_sizes = Counter(docids)
    for line, docid in zip(full_corpus, docids):
        if docid != prev_docid:
            doc_sent = 0
        else:
            doc_sent += 1

        prev_docid = docid
        if doc_sent < context_size - n_sentence:
            continue

        sentences = line.split("<brk>")
        # if it's the last sentence, and n_sentence < context_size,
        # we take extra sentences at the end
        if doc_sent == docs_sizes[docid] - 1:
            for n in range(n_sentence, context_size + 1):
                s = sentences[n].strip() if len(sentences) > n else ""
                corpus.append(s)
        # else we take take just the n_sentence, except in the
        # first sentences of the document, where we take the
        # last
        else:
            n = min(doc_sent, n_sentence)
            s = sentences[n].strip() if len(sentences) > n else ""
            corpus.append(s)

    return corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp")
    parser.add_argument("ref")
    parser.add_argument("--src", type=str, default=None)
    parser.add_argument("--docids", type=str, default=None)
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--speakers-file", type=str, default=None)
    parser.add_argument("--context-size", type=int, default=0)
    parser.add_argument("--n-sentence", type=int, default=None)
    parser.add_argument(
        "--comet-model",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--comet-path",
        default=None,
        type=str,
    )
    args = parser.parse_args()
    with open(args.hyp) as hyp_f:
        hyps = [line.strip() for line in hyp_f.readlines()]
    with open(args.ref) as ref_f:
        refs = [line.strip() for line in ref_f.readlines()]

    if args.speaker is not None:
        assert (
            args.speakers_file is not None
        ), "speaker file needed when evaluating performance for particular speaker"

        with open(args.speakers_file, "r") as speakers_f:
            speakers = [line.strip() for line in speakers_f.readlines()]

        new_refs = []
        for ref, spk in zip(refs, speakers):
            if spk == args.speaker:
                new_refs.append(ref)
        refs = new_refs

    if args.context_size > 0:
        assert args.docids is not None, "docids file needed when context size > 0"

        with open(args.docids) as docids_f:
            docids = [int(line) for line in docids_f.readlines()]

        if args.n_sentence is None:
            args.n_sentence = args.context_size

        hyps = make_n_sentence_corpus(hyps, docids, args.context_size, args.n_sentence)

    assert len(hyps) == len(refs)

    print(f"example hyp: {hyps[-1]}")
    print(f"example ref: {refs[-1]}")
    print(f"sacrebleu: {sacrebleu.corpus_bleu(hyps, [refs]).format()}")

    blonde = BLONDE()
    print(f"blonde: {blonde.corpus_score([hyps], [[refs]])}")

    if args.comet_model is not None:
        assert (
            args.comet_path is not None
        ), "need to provide a path to download/load comet"

        assert args.src is not None, "source needs to be provided to use COMET"
        with open(args.src) as src_f:
            srcs = [line.strip() for line in src_f.readlines()]

        # download comet and load
        model_path = download_model(args.comet_model)
        comet_model = load_from_checkpoint(model_path)
        print("running comet evaluation....")
        comet_input = [
            {"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(srcs, hyps, refs)
        ]
        seg_scores, sys_score = comet_model.predict(comet_input, gpus=1, batch_size=8)
        print(sys_score)
        print(f"COMET = {sum(seg_scores)/len(seg_scores):.4f}")


if __name__ == "__main__":
    main()
