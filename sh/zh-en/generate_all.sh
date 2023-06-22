#!/bin/bash

set -e

src=zh
tgt=en
lang=${src}-${tgt}
dropout=0.2
model_size=base

root=/project/jonmay_231/linghaoj/reproduce
repo=/project/jonmay_231/linghaoj/concat-src-only
model=${root}/concat_models
data=${root}/data/${lang}
bin=${data}/bin
bin_shuf=${data}/bin-shuf
ckpt_path=${root}/ckpt1
script_path=${repo}/scripts
COMET=/project/jonmay_231/jacqueline/contextual-mt/preds/comet

cd $root
###############################################################################

# xfmr 
# echo "xfmr baseline generating..."
# ckpt=${root}/$ckptfile/xfm-${dropout}[${lang}]
# pred=$ckpt/result
# mkdir -p $pred
# bash sh/${lang}/generate.sh \
#     --a=xfmr \
#     --bin=$bin \
#     --data=$data \
#     --src=${src} --tgt=${tgt} \
#     --model=$model \
#     --ckpt=$ckpt \
#     --pred=$pred \
#     --checkpoint=$checkpointfile \
#     --COMET=$COMET

###############################################################################

# concat / mega
for a in mega
do
    for n in 0 1
    do
        for m in 0 1
        do
            
            ckpt=${ckpt_path}/${a}-${n}-${m}-${dropout}[${lang}]
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
                --repo=$repo \
                --ckpt=$ckpt \
                --pred=$pred \
                --checkpoint=checkpoint_best.pt \
                --COMET=$COMET
        done
    done

    ckpt=${ckpt_path}/${a}-1-1-${dropout}[${lang}]
    echo "loaded $ckpt"
    pred=$ckpt/result-shuf
    mkdir -p $pred
    bash sh/${lang}/generate.sh \
        --a=$a \
        --t=reg \
        --sf=yes \
        --bin=$bin_shuf \
        --data=$data \
        --src=${src} --tgt=${tgt} \
        --repo=$repo \
        --ckpt=$ckpt \
        --pred=$pred \
        --checkpoint=checkpoint_best.pt \
        --COMET=$COMET

    ckpt=${ckpt_path}/${a}-src3-${dropout}[${lang}][new]
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
        --repo=$repo \
        --ckpt=$ckpt \
        --pred=$pred \
        --checkpoint=checkpoint_best.pt \
        --COMET=$COMET

    ckpt=${ckpt_path}/${a}-src3-${dropout}[${lang}][new]
    echo "loaded $ckpt"
    pred=$ckpt/result-shuf
    mkdir -p $pred
    bash sh/${lang}/generate.sh \
        --a=$a \
        --t=src3 \
        --sf=yes \
        --bin=$bin_shuf \
        --data=$data \
        --src=${src} --tgt=${tgt} \
        --repo=$repo \
        --ckpt=$ckpt \
        --pred=$pred \
        --checkpoint=checkpoint_best.pt \
        --COMET=$COMET
done
################################################################