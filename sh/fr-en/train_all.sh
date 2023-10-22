#!/bin/bash

set -e
cd /project/jonmay_231/linghaoj/reproduce

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

src=fr
tgt=en
lang=${src}-${tgt}
root=/project/jonmay_231/linghaoj/canmt-challenges
bin=${root}/data/$lang/bin
repo=${root}/concat_models
ckpt_path=${root}/ckpt
dropout=0.2
total_num_update=20000
wandb_project=test
model_size=base

cd $root

###############################################################################

for seed in 42 22 73
do
    # xfmr baseline
    ckpt=${ckpt_path}/xfmr-${dropout}-${model_size}-${seed}[${lang}]
    echo $ckpt
    mkdir -p $ckpt
    sbatch sh/${lang}/train.sh \
        --a=xfmr \
        --model_size=$model_size \
        --src=$src --tgt=$tgt \
        --bin=$bin --repo=$repo --ckpt=$ckpt \
        --dropout=$dropout \
        --total_num_update=$total_num_update \
        --seed=$seed \
        --wandb_project=$wandb_project

    # mega baseline
    ckpt=${ckpt_path}/mega-${dropout}-${model_size}-${seed}[${lang}]
    echo $ckpt
    mkdir -p $ckpt
    sbatch sh/${lang}/train.sh \
        --a=mega \
        --model_size=$model_size \
        --src=$src --tgt=$tgt \
        --bin=$bin --repo=$repo --ckpt=$ckpt \
        --total_num_update=$total_num_update \
        --dropout=$dropout \
        --seed=$seed \
        --wandb_project=$wandb_project

    # concat / concat-mega
    for a in concat concat-mega
    do
        ckpt=${ckpt_path}/${a}-src3-${dropout}-${model_size}-${seed}[${lang}]
        echo $ckpt
        mkdir -p $ckpt
        sbatch sh/${lang}/train.sh \
            --a=$a \
            --t=src3 \
            --s=nonsf \
            --src=$src --tgt=$tgt \
            --model_size=$model_size \
            --bin=$bin --repo=$repo --ckpt=$ckpt \
            --total_num_update=$total_num_update \
            --dropout=$dropout \
            --seed=$seed \
            --wandb_project=$wandb_project
    done
done