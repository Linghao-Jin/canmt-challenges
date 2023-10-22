checkpoint_dir='/mmfs1/gscratch/zlab/jyyh/canmt-challenges/ckpt/concat-contextual_transformer-fr-en-0-0-0.1'
src=fr
tgt=en
data_dir=/gscratch/zlab/jyyh/canmt-challenges/canmt-data/iwslt/$src-$tgt
predictions_dir=${checkpoint_dir}/out
COMET_DIR=/tmp/comet

python scripts/score.py ${predictions_dir}/test.pred.${tgt}  ${data_dir}/raw/test.${src}-${tgt}.${tgt} \
    --src ${data_dir}/raw/test.${src}-${tgt}.${src} \
    --comet-model wmt20-comet-da \
    --comet-path $COMET_DIR