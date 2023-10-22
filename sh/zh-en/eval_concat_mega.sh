src=zh
tgt=en
dp=0.3
rs=$1
# checkpoint_dir=/mmfs1/gscratch/zlab/jyyh/canmt-challenges/ckpt/3-1-concat_mega-${src}-${tgt}-dp-${dp}-rs-${rs}
checkpoint_dir=/mmfs1/gscratch/zlab/jyyh/canmt-challenges/ckpt/concat_mega-${src}-${tgt}-0-0-dp-${dp}-rs-${rs}

data_dir=/gscratch/zlab/jyyh/canmt-challenges/canmt-data/data/$src-$tgt
predictions_dir=${checkpoint_dir}/out
COMET_DIR=/tmp/comet

python scripts/score.py ${predictions_dir}/test.pred.${tgt}  ${data_dir}/raw/test.${src}-${tgt}.${tgt} \
    --src ${data_dir}/raw/test.${src}-${tgt}.${src} \
    --comet-model wmt20-comet-da \
    --comet-path $COMET_DIR