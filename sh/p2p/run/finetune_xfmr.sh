#!/bin/bash

#SBATCH --output=/project/jonmay_231/linghaoj/canmt-challenges/slurm/xfmr-wmt17-p2p.out
#SBATCH --error=/project/jonmay_231/linghaoj/canmt-challenges/slurm/xfmr-wmt17-p2p.err
#SBATCH --job-name=p2p-xfmr-finetune
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
    -s ${src} -t ${tgt} \
    --arch $arch \
    --optimizer adam  --lr 0.0001 --lr-scheduler inverse_sqrt --adam-betas '(0.9, 0.98)' \
    --dropout 0.2 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion label_smoothed_cross_entropy  --label-smoothing 0.1 \
    --max-update 40000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --seed ${seed} \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --keep-last-epochs 5 \
    --restore-file ${model}/checkpoint_best.pt \
    --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
    --max-source-positions 4096 --max-target-positions	4096 --max-tokens 4000 \
    --keep-last-epochs 5 \
    --fp16 \
    --wandb-project ${wandb_project}