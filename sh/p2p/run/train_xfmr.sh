#!/bin/bash

#SBATCH --output=/project/jonmay_231/linghaoj/reproduce/slurm/xfmr-wmt17-big.out
#SBATCH --error=/project/jonmay_231/linghaoj/reproduce/slurm/xfmr-wmt17-big.err
#SBATCH --job-name=bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=isi
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
#SBATCH --open-mode=append
#SBATCH --time=14-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=linghaojin@gmail.com

src='zh'
tgt='en'
BOOK=wmt17
ROOT=/project/jonmay_231/linghaoj/reproduce
DATA=$ROOT/data/p2p/$BOOK
BIN=$DATA/bin-wmt17
CKPT=$DATA/ckpt/xfmr-wmt17-big

# rm -rf $CKPT
mkdir -p $CKPT

srun --label fairseq-train ${BIN} \
    -s ${src} -t ${tgt} \
    --arch transformer_wmt_en_de_big --save-dir ${CKPT} \
    --optimizer adam  --lr 0.001 --adam-betas '(0.9, 0.98)'  \
    --lr-scheduler inverse_sqrt --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy  --label-smoothing 0.1 \
    --warmup-updates 10000 --warmup-init-lr '1e-07' \
    --max-tokens 4000 --update-freq 16 \
    --max-update 2000000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --keep-last-epochs 20 \
    --fp16 \
    --max-source-positions	1024 \
    --max-target-positions	1024 \
    --wandb-project reproduce-p2p