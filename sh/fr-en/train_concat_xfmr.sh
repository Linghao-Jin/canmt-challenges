#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --mail-user=jyyh@uw.edu
#SBATCH --partition=gpu-a40
#SBATCH --account=zlab
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gpus=4
#SBATCH --job-name=repro-concat-xfmr
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jacquelinehe00@gmail.com
#SBATCH -o slurm/slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e slurm/slurm-%j.err-%N # name of the stderr, using job and first node values

src=fr
tgt=en
N=0
M=0
dp=0.3
seed=35
model=contextual_transformer
TOTAL_NUM_UPDATES=200000
BIN=/gscratch/zlab/jyyh/canmt-challenges/canmt-data/iwslt/$src-$tgt/bin
REPO=/gscratch/zlab/jyyh/canmt-challenges/concat_models
CKPT=/gscratch/zlab/jyyh/canmt-challenges/ckpt/concat-${model}-$src-$tgt-$N-$M-dp-${dp}-rs-${seed}

fairseq-train ${BIN} \
    --user-dir ${REPO} --save-dir ${CKPT} \
	--task document_translation -s ${src} -t ${tgt} \
    --source-context-size $N --target-context-size $M \
    --arch ${model} --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
    --lr 5e-4 --lr-scheduler inverse_sqrt  --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout $dp --weight-decay 0.0001 \
    --max-tokens 4096 --update-freq 8 --patience 10 --seed ${seed} \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --fp16 --no-epoch-checkpoints