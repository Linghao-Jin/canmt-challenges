#!/bin/bash

#SBATCH --output=/project/jonmay_231/linghaoj/canmt-challenges/slurm/generate_zh.out
#SBATCH --error=/project/jonmay_231/linghaoj/canmt-challenges/slurm/generate_zh.err
#SBATCH --job-name=gen-zh
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2
#SBATCH --partition=isi
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
#SBATCH --open-mode=truncate
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=linghaojin@gmail.com
set -e

src=zh
tgt=en
lang=${src}-${tgt}
dropout=0.2
model_size=base

root=/project/jonmay_231/linghaoj/canmt-challenges
model=${root}/concat_models
data=${root}/data/${lang}
bin=${data}/bin
ckpt_path=${root}/ckpt
ckpt_name=checkpoint_best.pt
# ckpt_name=checkpoint_last5_avg.pt
script_path=${root}/scripts
COMET=/project/jonmay_231/jacqueline/contextual-mt/preds/comet

cd $root

for seed in 22 42 73
do
    ###############################################################################

    # xfmr 
    ckpt=${ckpt_path}/xfmr-${dropout}-${seed}[${lang}]
    echo "loaded $ckpt"
    pred=$ckpt/result
    mkdir -p $pred
    bash sh/${lang}/generate.sh \
        --a=xfmr \
        --bin=$bin \
        --data=$data \
        --src=${src} --tgt=${tgt} \
        --model=$model \
        --ckpt=$ckpt \
        --pred=$pred \
        --checkpoint=$ckpt_name \
        --COMET=$COMET

    ###############################################################################

    # concat / concat-mega
    for a in concat-mega
    do
        for n in 0 1
        do
            for m in 0 1
            do
                if [ "$n" -ne 0 ] || [ "$m" -ne 0 ]; 
                then
                    ckpt=${ckpt_path}/${a}-${n}-${m}-${dropout}-${seed}[${lang}]
                    echo "loaded $ckpt"
                    pred=$ckpt/result
                    mkdir -p $pred
                    bash sh/${lang}/generate.sh \
                        --a=$a \
                        --t=reg \
                        --sf=no \
                        --bin=$bin \
                        --data=$data \
                        --src=${src} --tgt=${tgt} \
                        --repo=$root \
                        --ckpt=$ckpt \
                        --pred=$pred \
                        --checkpoint=$ckpt_name \
                        --COMET=$COMET
                fi
            done
        done

        ckpt=${ckpt_path}/${a}-src3-${dropout}-${seed}[${lang}]
        echo "loaded $ckpt"
        pred=$ckpt/result
        mkdir -p $pred
        bash sh/${lang}/generate.sh \
            --a=$a \
            --t=src3 \
            --sf=no \
            --bin=$bin \
            --data=$data \
            --src=${src} --tgt=${tgt} \
            --repo=$root \
            --ckpt=$ckpt \
            --pred=$pred \
            --checkpoint=$ckpt_name \
            --COMET=$COMET
    done
done