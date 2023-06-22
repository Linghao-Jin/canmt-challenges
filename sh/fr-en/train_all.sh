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
root=/project/jonmay_231/linghaoj/reproduce
bin=${root}/data/fr-en/bin
repo=${root}/concat_models
ckpt_path=${root}/ckpt3
dropout=0.3
attention_dropout=0.1
activation_dropout=0.1
total_num_update=10000
wandb_project=reproduce-mega
model_size=base
###############################################################################


# xfmr 
# echo "xfm-${dropout}-${model_size}[${src}-${tgt}] running..."
# ckpt=${ckpt_path}/xfm-${dropout}-${model_size}[${src}-${tgt}]
# mkdir -p $ckpt
# sbatch sh/${src}-${tgt}/train.sh \
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
    for n in 0
    do
        for m in  0
        do
            update_freq=8
            
            if [ $n = 0 ] && [ $m = 0 ];
            then 
                update_freq=4
            fi
            
            ckpt=${ckpt_path}/${a}-${n}-${m}-${dropout}-${model_size}[${src}-${tgt}][halfbsz]
            echo "loaded $ckpt"
            mkdir -p $ckpt
            sbatch sh/${src}-${tgt}/train.sh \
                --a=$a \
                --t=reg \
                --s=nonsf \
                --src=$src --tgt=$tgt \
                --n=$n --m=$m \
                --model_size=$model_size \
                --bin=$bin --repo=$repo --ckpt=$ckpt \
                --total_num_update=$total_num_update \
                --attention-dropout=$attention_dropout \
                --activation_dropout=$activation_dropout \
                --update_freq=$update_freq \
                --wandb_project=$wandb_project
        done
    done

    # echo "${a}-1-1-sf running"
    # ckpt=${ckpt_path}/${a}-1-1-${dropout}-${model_size}-sf[${src}-${tgt}]
    # mkdir -p $ckpt
    # sbatch sh/${src}-${tgt}/train.sh \
    #     --a=$a \
    #     --t=reg \
    #     --s=sf \
    #     --src=$src --tgt=$tgt \
    #     --n=1 --m=1 \
    #     --model_size=$model_size \
    #     --bin=$bin --repo=$repo --ckpt=$ckpt \
    #     --total_num_update=$total_num_update \
    #     --wandb_project=$wandb_project

    # update_freq=12
    # echo "${a}-src3-${dropout}-${model_size}[${src}-${tgt}] running" 
    # ckpt=${ckpt_path}/${a}-src3-${dropout}-${model_size}[${src}-${tgt}]
    # mkdir -p $ckpt
    # sbatch sh/${src}-${tgt}/train.sh \
    #     --a=$a \
    #     --t=src3 \
    #     --s=nonsf \
    #     --src=$src --tgt=$tgt \
    #     --model_size=$model_size \
    #     --bin=$bin --repo=$repo --ckpt=$ckpt \
    #     --total_num_update=$total_num_update \
    #     --attention-dropout=$attention_dropout \
    #     --activation_dropout=$activation_dropout \
    #     --update_freq=$update_freq \
    #     --wandb_project=$wandb_project

    # echo "${a}-src3-${dropout}-sf running"
    # ckpt=${ckpt_path}/${a}-src3-${dropout}-${model_size}-sf[${src}-${tgt}]
    # mkdir -p $ckpt
    # sbatch sh/${src}-${tgt}/train.sh \
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