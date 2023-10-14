#!/bin/bash

set -e

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

src=de
tgt=en
lang=${src}-${tgt}
root=/project/jonmay_231/linghaoj/canmt-challenges
bin=${root}/data/$lang/bin
repo=${root}/concat_models
ckpt_path=${root}/ckpt
dropout=0.3
attention_dropout=0.1
activation_dropout=0.1
total_num_update=10000
wandb_project=test
model_size=small

cd $root

###############################################################################

for seed in 42 22 73
do
    # xfmr 
    ckpt=${ckpt_path}/xfmr-${dropout}-${model_size}-${seed}[${lang}]
    echo $ckpt
    mkdir -p $ckpt
    sbatch sh/${lang}/train.sh \
        --a=xfmr \
        --model_size=$model_size \
        --src=$src --tgt=$tgt \
        --bin=$bin --repo=$repo --ckpt=$ckpt \
        --total_num_update=$total_num_update \
        --seed=$seed \
        --wandb_project=$wandb_project

    # concat / mega
    for a in concat
    do
        # for n in 0 1
        # do
        #     for m in  0 1
        #     do
        #         update_freq=8
                
        #         if [ $n = 0 ] && [ $m = 0 ];
        #         then 
        #             update_freq=4
        #         else
        #             update_freq=8

        #             ckpt=${ckpt_path}/${a}-${n}-${m}-${dropout}-${model_size}-${seed}[${lang}]
        #             echo $ckpt
        #             mkdir -p $ckpt
        #             sbatch sh/${lang}/train.sh \
        #                 --a=$a \
        #                 --t=reg \
        #                 --s=nonsf \
        #                 --src=$src --tgt=$tgt \
        #                 --n=$n --m=$m \
        #                 --model_size=$model_size \
        #                 --bin=$bin --repo=$repo --ckpt=$ckpt \
        #                 --total_num_update=$total_num_update \
        #                 --attention-dropout=$attention_dropout \
        #                 --activation_dropout=$activation_dropout \
        #                 --update_freq=$update_freq \
        #                 --seed=$seed \
        #                 --wandb_project=$wandb_project
        #         fi
        #     done
        # done
        
        # update_freq=12
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
            --attention-dropout=$attention_dropout \
            --activation_dropout=$activation_dropout \
            --seed=$seed \
            --wandb_project=$wandb_project

    done
done