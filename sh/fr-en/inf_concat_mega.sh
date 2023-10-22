src=fr
tgt=en
dp=0.3
rs=$1
#checkpoint_dir=/mmfs1/gscratch/zlab/jyyh/canmt-challenges/ckpt/3-1-concat_mega-${src}-${tgt}-dp-${dp}-rs-${rs}
checkpoint_dir=/mmfs1/gscratch/zlab/jyyh/canmt-challenges/ckpt/concat_mega-${src}-${tgt}-0-0-dp-${dp}-rs-${rs}

data_dir=/gscratch/zlab/jyyh/canmt-challenges/canmt-data/iwslt/$src-$tgt
predictions_dir=${checkpoint_dir}/out
mkdir -p ${predictions_dir}
echo ${checkpoint_dir}
cp ${data_dir}/bin/dict.*txt ${data_dir}/prep/spm* $checkpoint_dir

python docmt_translate.py \
    --path $checkpoint_dir \
    --source-lang ${src} --target-lang ${tgt} \
    --source-file ${data_dir}/raw/test.${src}-${tgt}.${src} \
    --predictions-file ${predictions_dir}/test.pred.${tgt} \
    --docids-file ${data_dir}/raw/test.${src}-${tgt}.docids \
    --beam 5 