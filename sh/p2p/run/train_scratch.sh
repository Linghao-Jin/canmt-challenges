#!/bin/bash

#SBATCH --output=/project/jonmay_231/linghaoj/canmt-challenges/slurm/sample-scratch-wmt17.out
#SBATCH --error=/project/jonmay_231/linghaoj/canmt-challenges/slurm/sample-scratch-wmt17.err
#SBATCH --job-name=p2p-scratch
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2
#SBATCH --partition=isi
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
#SBATCH --open-mode=truncate
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=linghaojin@gmail.com

# Read arguments
for argument in "$@" 
do
    key=$(echo $argument | cut -f1 -d=)
    value=$(echo $argument | cut -f2 -d=)
    # Use string manipulation to set variable names according to convention   
    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        v="${v/-/_}"
        declare $v="${value}" 
   fi
done

src=zh
tgt=en

mkdir -p $ckpt

fairseq-train ${bin} --save-dir ${ckpt} \
    --task translation -s ${src} -t ${tgt} \
    --arch $arch \
    --optimizer adam  --lr 0.0001 --lr-scheduler inverse_sqrt --adam-betas '(0.9, 0.98)' \
    --dropout 0.2 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion label_smoothed_cross_entropy  --label-smoothing 0.1 \
    --max-update 40000  --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --seed ${seed} \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5}' \
    --eval-bleu-remove-bpe \
    --eval-bleu-detok moses \
    --keep-last-epochs 5 \
    --max-source-positions 4096 --max-target-positions	4096 --max-tokens 4000 \
    --fp16 \
    --wandb-project ${wandb_project}