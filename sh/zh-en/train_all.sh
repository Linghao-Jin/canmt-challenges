#!/bin/bash

set -e

src=zh
tgt=en
lang=${src}-${tgt}
root=/project/jonmay_231/linghaoj/canmt-challenges
bin=${root}/data/${lang}/bin
repo=${root}/concat_models
ckpt_path=${root}/ckpt
dropout=0.2
total_num_update=300000
wandb_project=camera-ready
model_size=base

cd $root

###############################################################################

for seed in 42 22 73
do
    ## xfmr 
    ckpt=${ckpt_path}/xfmr-${dropout}[${lang}]
    echo $ckpt
    mkdir -p $ckpt
    sbatch sh/${lang}/train.sh \
        --a=xfmr \
        --model_size=$model_size \
        --src=$src --tgt=$tgt \
        --bin=$bin --repo=$repo --ckpt=$ckpt \
        --total_num_update=$total_num_update \
        --wandb_project=$wandb_project

    # concat / mega
    for a in concat
    do
        for n in 0 1
        do
            for m in 0 1
            do
                if [ "$n" -ne 0 ] || [ "$m" -ne 0 ]; 
                then
                    ckpt=${ckpt_path}/${a}-${n}-${m}-${dropout}[${lang}]
                    echo $ckpt
                    mkdir -p $ckpt
                    sbatch sh/${lang}/train.sh \
                        --a=$a \
                        --t=reg \
                        --s=nonsf \
                        --src=$src --tgt=$tgt \
                        --n=$n --m=$m \
                        --model_size=$model_size \
                        --bin=$bin --repo=$repo --ckpt=$ckpt \
                        --total_num_update=$total_num_update \
                        --wandb_project=$wandb_project
                fi
            done
        done

    ckpt=${ckpt_path}/${a}-src3-${dropout}[${lang}]
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
        --wandb_project=$wandb_project

    done
done