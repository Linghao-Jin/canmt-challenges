#!/bin/bash

set -e

src=zh
tgt=en
lang=${src}-${tgt}
root=/project/jonmay_231/linghaoj/canmt-challenges
# wandb_project=camera-ready-p2p
wandb_project=camera-ready-p2p


cd $root

###############################################################################

for seed in 42 22 73
do
    for book in samples samples-sf
    do
        # train from scratch 
        bin=${root}/data/p2p/$book/bin-wmt17
        ckpt=${root}/ckpt-p2p/$book/xfmr-scratch-p2p-$seed
        echo $ckpt
        arch=transformer_vaswani_wmt_en_de_big
        # arch=transformer 
        sbatch sh/p2p/run/train_scratch.sh \
            --bin=$bin --ckpt=$ckpt --arch=$arch \
            --seed=$seed \
            --wandb_project=$wandb_project
        
        
        ####################################################
        # finetune from transformer-wmt17
        bin=${root}/data/p2p/$book/bin-custom-wmt17
        ckpt=${root}/ckpt-p2p/$book/xfmr-big-wmt17-p2p-$seed
        echo $ckpt
        model=${root}/ckpt-p2p/wmt17/xfmr-big-wmt17
        arch=transformer_vaswani_wmt_en_de_big
        # arch=transformer 

        sbatch sh/p2p/run/finetune_xfmr.sh \
            --bin=$bin --ckpt=$ckpt --model=$model --arch=$arch \
            --seed=$seed \
            --wandb_project=$wandb_project


        ###################################################
        # finetune from lightconv-wmt17
        bin=${root}/data/p2p/$book/bin-custom-wmt17
        ckpt=${root}/ckpt-p2p/$book/lightconv-wmt17-p2p-$seed
        echo $ckpt
        model=${root}/ckpt-p2p/wmt17/lightconv-wmt17
        arch=lightconv_wmt_zh_en_big

        sbatch sh/p2p/run/finetune_lightconv.sh \
                --bin=$bin --ckpt=$ckpt --model=$model --arch=$arch \
                --seed=$seed \
                --wandb_project=$wandb_project


        ####################################################
        # finetune from mbart-wmt17
        bin=${root}/data/p2p/$book/bin-mbart
        ckpt=${root}/ckpt-p2p/$book/mbart-wmt17-p2p-$seed
        echo $ckpt
        model=${root}/ckpt-p2p/wmt17/mbart-zh-en

        sbatch sh/p2p/run/finetune_mbart.cc25.sh \
            --bin=$bin --ckpt=$ckpt --model=$model \
            --seed=$seed \
            --wandb_project=$wandb_project

    done
done

