#!/bin/bash

# USAGE ./prepare_shuffle_test.sh src_lang (e.g. de)
src=$1
tgt=en
lang=${src}-${tgt}
repo=/project/jonmay_231/linghaoj/reproduce
data_dir=$repo/data/$lang
raw_dir=$data_dir/raw
prep_dir=$data_dir/prep

cd $repo


# generate shuffled test files
echo "generate shuffled test files...."
bash scripts/shuffle_multiple_files.sh $raw_dir/test.$lang.$src $raw_dir/test.$lang.$tgt $raw_dir/test.$lang.docids

echo "spm encoding..."
for l in $src $tgt; do
    python scripts/spm_encode.py \
        --model $prep_dir/spm.$l.model \
        < ${raw_dir}/test.$lang.$l.shuf \
        > ${prep_dir}/test.shuf.sp.${l}
done

echo "fairseq preprocessing..."
rm -rf $data_dir/bin-shuf
fairseq-preprocess \
    --source-lang $src --target-lang $tgt \
    --srcdict $data_dir/bin/dict.$src.txt --tgtdict $data_dir/bin/dict.$tgt.txt \
    --trainpref $prep_dir/train.sp \
    --validpref $prep_dir/valid.sp \
    --testpref $prep_dir/test.shuf.sp \
    --destdir $data_dir/bin-shuf