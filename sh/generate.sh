#!/bin/bash

set -e

####################################################################################
# custom settings
src=de
tgt=en
lang=${src}-${tgt}
ckptfile=ckpt2
a=mega # mega xfmr
t=reg # src
sf=no
dp=0.4
model_size=base
ckptname=$a-0-1-$dp-$model_size[$lang]
checkpointfile=checkpoint_best.pt

####################################################################################
root=/project/jonmay_231/linghaoj/reproduce
repo=/project/jonmay_231/linghaoj/concat-src-only
model=${repo}/concat_models
data=${root}/data/${lang}
ckpt=${root}/$ckptfile/$ckptname

if [[ $sf = "yes" ]]
then
    bin=${data}/bin-shuf # enable shuffled test set
    pred=$ckpt/result-shuf
else
    bin=${data}/bin
    pred=$ckpt/result
fi 
script_path=${repo}/scripts
COMET=/project/jonmay_231/jacqueline/contextual-mt/preds/comet

cd $root

echo "loaded $ckpt , shuffle setting is: $sf"
mkdir -p $pred
bash sh/${lang}/generate.sh \
    --a=$a \
    --t=$t \
    --sf=$sf \
    --bin=$bin \
    --data=$data \
    --src=${src} --tgt=${tgt} \
    --repo=$repo \
    --ckpt=$ckpt \
    --pred=$pred \
    --checkpoint=$checkpointfile \
    --COMET=$COMET