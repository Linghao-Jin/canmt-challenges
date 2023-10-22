#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=jyyh@uw.edu
#SBATCH --partition=gpu-a40
#SBATCH --account=zlab
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=4
#SBATCH --job-name=repro-zh-concat-mega
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jacquelinehe00@gmail.com
#SBATCH -o slurm/slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e slurm/slurm-%j.err-%N # name of the stderr, using job and first node values

seed=$1
src=zh
tgt=en
N=1
M=1
dp=0.3
model=concat_mega
TOTAL_NUM_UPDATES=200000
BIN=/gscratch/zlab/jyyh/canmt-challenges/canmt-data/data/$src-$tgt/bin
REPO=/gscratch/zlab/jyyh/canmt-challenges/concat_models
CKPT=/gscratch/zlab/jyyh/canmt-challenges/ckpt/${model}-$src-$tgt-$N-$M-dp-${dp}-rs-${seed}


fairseq-train ${BIN} \
    --user-dir ${REPO} --save-dir ${CKPT} \
	--task document_translation -s ${src} -t ${tgt} \
    --source-context-size $N --target-context-size $M \
    --arch ${model} --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1 \
    --lr 1e-3 --lr-scheduler inverse_sqrt  --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout $dp --weight-decay 0.01 \
    --max-tokens 4096 --update-freq 8 --save-interval-updates 5000 --max-update 500000  --seed ${seed} \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --keep-last-epochs 3 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --fp16 --no-epoch-checkpoints

# fairseq-train ${BIN} \
#     --user-dir ${REPO} --save-dir ${CKPT} \
# 	--task document_translation -s ${src} -t ${tgt} \
#     --source-context-size $N --target-context-size $M \
#     --arch ${model} --encoder-layers 6 --decoder-layers 6 --share-decoder-input-output-embed \
#     --activation-fn silu --attention-activation-fn softmax \
#     --encoder-n-dim 16 --encoder-chunk-size -1 \
#     --normalization-type layernorm \
#     --ddp-backend c10d \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 0.1 \
#     --lr 1e-3 --lr-scheduler inverse_sqrt  --warmup-updates 4000 \
#     --dropout $dp --attention-dropout 0.0 --hidden-dropout 0.0 --activation-dropout 0.0 --weight-decay 0.01 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 4096 --update-freq 16 --seed 42 --max-update ${TOTAL_NUM_UPDATES} \
#     --eval-bleu --eval-bleu-print-samples --eval-bleu-remove-bpe sentencepiece \
#     --eval-bleu-args '{"beam": 5}' \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#     --keep-last-epochs 5 \
#     --validate-after-updates 5 \
#     --fp16