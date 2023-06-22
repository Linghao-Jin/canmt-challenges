#!/bin/bash

#SBATCH --output=/project/jonmay_231/linghaoj/reproduce/slurm/xfm-src3-sf-0.2
#SBATCH --error=/project/jonmay_231/linghaoj/reproduce/slurm/xfm-src3-sf-0.2
#SBATCH --job-name=bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
#SBATCH --open-mode=append
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=linghaojin@gmail.com


src=zh
tgt=en
dp=0.2
BIN=/project/jonmay_231/linghaoj/canmt/bwb/data/bin
REPO=/project/jonmay_231/linghaoj/concat-src-only/concat_models
CKPT=/project/jonmay_231/linghaoj/reproduce/ckpt/xfm-src3-${dp}-sf[zh-en]
mkdir -p ${CKPT}

srun --label fairseq-train ${BIN} \
    --user-dir ${REPO} --save-dir ${CKPT} \
	--task document_translation -s ${src} -t ${tgt} \
    --arch contextual_transformer --share-decoder-input-output-embed \
    --next-sent-ctx \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1 \
    --lr 1e-3 --lr-scheduler inverse_sqrt  --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout $dp --weight-decay 0.01 \
    --max-tokens 2048 --update-freq 16 --seed 42 --max-update 300000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --keep-last-epochs 5 \
    --fp16 \
    --shuffle_sample \
    --wandb-project reproduce-doc-mt
