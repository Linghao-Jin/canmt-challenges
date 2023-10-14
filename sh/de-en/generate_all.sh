#!/bin/bash

#SBATCH --output=/project/jonmay_231/linghaoj/reproduce/slurm/generate_de.out
#SBATCH --error=/project/jonmay_231/linghaoj/reproduce/slurm/generate_de.err
#SBATCH --job-name=gen-de
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
#SBATCH --open-mode=append
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=linghaojin@gmail.com

src=de
tgt=en
lang=${src}-${tgt}
dropout=0.3
model_size=small

root=/project/jonmay_231/linghaoj/canmt-challenges
model=${root}/concat_models
data=${root}/data/${lang}
bin=${data}/bin
bin_shuf=${data}/bin-shuf
ckpt_path=${root}/ckpt
ckpt_name=checkpoint_best.pt
COMET=/project/jonmay_231/jacqueline/contextual-mt/preds/comet

cd $root

for seed in 22 42 73
do
    ###############################################################################

    # xfmr 
    ckpt=${ckpt_path}/xfmr-${dropout}-${model_size}-${seed}[${lang}]
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

    # concat / mega
    for a in concat
    do
  
        ckpt=${ckpt_path}/${a}-src3-${dropout}-${model_size}-${seed}[${lang}]
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
        
        # for n in 0 1
        # do
        #     for m in 0 1
        #     do
                
        #         ckpt=${ckpt_path}/${a}-${n}-${m}-${dropout}-${model_size}-${seed}[${lang}]
        #         echo "loaded $ckpt"
        #         pred=$ckpt/result
        #         mkdir -p $pred
        #         bash sh/${lang}/generate.sh \
        #             --a=$a \
        #             --t=reg \
        #             --sf=no \
        #             --bin=$bin \
        #             --data=$data \
        #             --src=${src} --tgt=${tgt} \
        #             --repo=$root \
        #             --ckpt=$ckpt \
        #             --pred=$pred \
        #             --checkpoint=checkpoint_best.pt \
        #             --COMET=$COMET
        #     done
        # done
    done
    ###############################################################################
done