#!/bin/bash

#SBATCH --output=/project/jonmay_231/linghaoj/reproduce/slurm/concat-0-0-0.2
#SBATCH --error=/project/jonmay_231/linghaoj/reproduce/slurm/concat-0-0-0.2
#SBATCH --job-name=zh-tmp
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=isi
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
#SBATCH --open-mode=append
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=linghaojin@gmail.com

N=0
M=1
dp=0.2 # don't forget to change 
src=zh
tgt=en
updates=300000
CKPT=/project/jonmay_231/linghaoj/reproduce/ckpt1/concat-$N-$M-${dp}[zh-en][tmp-8192] # don't forget change ckpt
REPO=/project/jonmay_231/linghaoj/concat-src-only/concat_models
BIN=/project/jonmay_231/linghaoj/canmt/bwb/data/bin

mkdir -p ${CKPT}

srun --label fairseq-train \
	${BIN} --user-dir ${REPO} --save-dir ${CKPT} \
	--task document_translation -s ${src} -t ${tgt}\
    --arch contextual_transformer --share-decoder-input-output-embed \
    --source-context-size ${N} --target-context-size ${M} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
    --lr 1e-3 --lr-scheduler inverse_sqrt  --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --dropout $dp --coword-dropout 0 --weight-decay 0.01 \
    --max-tokens 8192 --update-freq 16 --seed 42 --max-update $updates \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5}' \
    --eval-bleu-remove-bpe sentencepiece \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --keep-last-epochs 5 \
    --fp16 \
    --wandb-project reproduce-doc-mt-zh
    # --sample-context-size \
    # --shuffle_sample \
