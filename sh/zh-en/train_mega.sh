#!/bin/bash

#SBATCH --output=/project/jonmay_231/linghaoj/reproduce/slurm/mega-src3.out
#SBATCH --error=/project/jonmay_231/linghaoj/reproduce/slurm/mega-src3.err
#SBATCH --job-name=mega
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
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
N=1
M=1
dp=0.2
model=contextual_mega
TOTAL_NUM_UPDATES=200000

BIN=/project/jonmay_231/linghaoj/canmt/bwb/data/bin
REPO=/project/jonmay_231/linghaoj/reproduce/concat_models
CKPT=/project/jonmay_231/linghaoj/reproduce/ckpt/mega-$N-$M-${dp}[zh-en][new]
mkdir -p ${CKPT}

srun --label fairseq-train ${BIN} \
    --user-dir ${REPO} --save-dir ${CKPT} \
	--task document_translation -s ${src} -t ${tgt} \
    --source-context-size $N --target-context-size $M \
    --arch ${model} --encoder-layers 6 --decoder-layers 6 --share-decoder-input-output-embed \
    --activation-fn silu --attention-activation-fn softmax \
    --encoder-n-dim 16 --encoder-chunk-size -1 \
    --normalization-type layernorm \
    --ddp-backend c10d \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 0.1 \
    --lr 1e-3 --lr-scheduler inverse_sqrt  --warmup-updates 4000 \
    --dropout $dp --attention-dropout 0.0 --hidden-dropout 0.0 --activation-dropout 0.0 --weight-decay 0.01 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --update-freq 16 --seed 42 --max-update ${TOTAL_NUM_UPDATES} \
    --eval-bleu --eval-bleu-print-samples --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-args '{"beam": 5}' \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --keep-last-epochs 5 \
    --fp16 \
    --wandb-project reproduce-doc-mt
    # --shuffle_sample \
    

