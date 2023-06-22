#!/bin/bash

set -e

src=zh
tgt=en
lang=${src}-${tgt}
root=/project/jonmay_231/linghaoj/reproduce
bin=${root}/data/${lang}/bin
repo=${root}/concat_models
ckpt_path=${root}/ckpt1
dropout=0.2
total_num_update=300000
wandb_project=reproduce-doc-mt-zh
model_size=base

cd $root
###############################################################################

# xfmr 
# ckpt=${ckpt_path}/xfm-${dropout}[${lang}]
# echo $ckpt
# mkdir -p $ckpt
# sbatch sh/${lang}/train.sh \
#     --a=xfmr \
#     --model_size=$model_size \
#     --src=$src --tgt=$tgt \
#     --bin=$bin --repo=$repo --ckpt=$ckpt \
#     --total_num_update=$total_num_update \
#     --wandb_project=$wandb_project

###############################################################################

# concat / mega
for a in mega
do
    # for n in 0 1
    # do
    #     for m in 0 1
    #     do
    #         ckpt=${ckpt_path}/${a}-${n}-${m}-${dropout}[${lang}]
    #         echo $ckpt
    #         mkdir -p $ckpt
    #         sbatch sh/${lang}/train.sh \
    #             --a=$a \
    #             --t=reg \
    #             --s=nonsf \
    #             --src=$src --tgt=$tgt \
    #             --n=$n --m=$m \
    #             --model_size=$model_size \
    #             --bin=$bin --repo=$repo --ckpt=$ckpt \
    #             --total_num_update=$total_num_update \
    #             --wandb_project=$wandb_project
    #     done
    # done

    # ckpt=${ckpt_path}/${a}-1-1-${dropout}-sf[${lang}][new]
    # echo $ckpt
    # mkdir -p $ckpt
    # sbatch sh/${lang}/train.sh \
    #     --a=$a \
    #     --t=reg \
    #     --s=sf \
    #     --src=$src --tgt=$tgt \
    #     --n=1 --m=1 \
    #     --model_size=$model_size \
    #     --bin=$bin --repo=$repo --ckpt=$ckpt \
    #     --total_num_update=$total_num_update \
    #     --wandb_project=$wandb_project

    ckpt=${ckpt_path}/${a}-src3-${dropout}[${lang}][new]
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

    # ckpt=${ckpt_path}/${a}-src3-${dropout}-sf[${lang}][new]
    # echo $ckpt
    # mkdir -p $ckpt
    # sbatch sh/${lang}/train.sh \
    #     --a=$a \
    #     --t=src3 \
    #     --s=sf \
    #     --src=$src --tgt=$tgt \
    #     --model_size=$model_size \
    #     --bin=$bin --repo=$repo --ckpt=$ckpt \
    #     --total_num_update=$total_num_update \
    #     --wandb_project=$wandb_project

done
###############################################################################