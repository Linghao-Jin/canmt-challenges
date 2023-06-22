#!/bin/bash

# This script provides command-line statements for training and testing a
# Transformer model for the translation of a concatenated sequence of
# source and/or target sentences, via the "mode" and "val" arguments

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

if [ -n "$checkpoint" ]; then checkpoint=$checkpoint ; else checkpoint=checkpoint_best.pt ; fi
if [ -n "$batch_size" ]; then batch_size=$batch_size ; else batch_size=128 ; fi
if [ -n "$beam" ]; then beam=$beam ; else beam=5 ; fi
if [ -n "$remove_bpe" ]; then remove_bpe=$remove_bpe ; else remove_bpe=sentencepiece ; fi
if [ -n "$script_path" ]; then script_path=$script_path ; else script_path=/project/jonmay_231/linghaoj/concat-src-only/scripts ; fi
if [ -n "$split" ]; then split=$split ; else split=test ; fi
if [ -n "$sf" ]; then sf=$sf ; else sf=no ; fi

testlog=best

cp ${data}/prep/dict.*txt ${data}/prep/spm* $ckpt

if [[ $sf == "yes" ]]
then 
    source_file=${data}/raw/${split}.${src}-${tgt}.${src}.shuf
    target_file=${data}/raw/${split}.${src}-${tgt}.${tgt}.shuf
    docids_file=${data}/raw/${split}.${src}-${tgt}.docids.shuf
else
    source_file=${data}/raw/${split}.${src}-${tgt}.${src}
    target_file=${data}/raw/${split}.${src}-${tgt}.${tgt}
    docids_file=${data}/raw/${split}.${src}-${tgt}.docids
fi

if [[ $a = "xfmr" ]]
then

    fairseq-generate ${bin} \
        --user-dir ${model} \
        --path ${ckpt}/${checkpoint} \
        --batch-size ${batch_size} \
        --beam ${beam} \
        --remove-bpe ${remove_bpe} \
        --scoring sacrebleu \
        > ${pred}/${testlog}.log

    grep ^T ${pred}/${testlog}.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- > ${pred}/${testlog}.ref
    grep ^H ${pred}/${testlog}.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- > ${pred}/${testlog}.sys
    grep ^S ${pred}/${testlog}.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- > ${pred}/${testlog}.src

    python ${script_path}/score.py \
        ${pred}/${testlog}.sys ${pred}/${testlog}.ref \
        --src ${pred}/${testlog}.src \
        --comet-model wmt20-comet-da \
        --comet-path ${COMET} > ${pred}/${testlog}.score

elif [[ $a = "concat" ]]
then
    if [[ $t = "src3" ]]
    then
        python ${repo}/docmt_translate.py \
        --path ${ckpt} \
        --source-lang ${src} --target-lang ${tgt} \
        --next-sent-ctx \
        --source-file $source_file \
        --predictions-file ${pred}/${testlog}.sys \
        --docids-file $docids_file \
        --beam $beam

    elif [[ $t = "reg" ]]
    then
        python ${repo}/docmt_translate.py \
        --path ${ckpt} \
        --source-lang ${src} --target-lang ${tgt} \
        --source-file $source_file \
        --predictions-file ${pred}/${testlog}.sys \
        --docids-file $docids_file \
        --beam $beam
    
    fi

    python ${script_path}/score.py ${pred}/${testlog}.sys $target_file \
        --src $source_file \
        --comet-model wmt20-comet-da \
        --comet-model wmt20-comet-da \
        --comet-path ${COMET} > ${pred}/${testlog}.score

elif [[ $a = "mega" ]]
then
    if [[ $t = "src3" ]]
    then
        python ${repo}/docmt_translate.py \
        --path ${ckpt} \
        --source-lang ${src} --target-lang ${tgt} \
        --next-sent-ctx \
        --source-file $source_file \
        --predictions-file ${pred}/${testlog}.sys \
        --docids-file $docids_file \
        --beam $beam

    elif [[ $t = "reg" ]]
    then
        python ${repo}/docmt_translate.py \
        --path ${ckpt} \
        --source-lang ${src} --target-lang ${tgt} \
        --source-file $source_file \
        --predictions-file ${pred}/${testlog}.sys \
        --docids-file $docids_file \
        --beam $beam
    
    fi

    python ${script_path}/score.py ${pred}/${testlog}.sys $target_file \
        --src $source_file \
        --comet-model wmt20-comet-da \
        --comet-model wmt20-comet-da \
        --comet-path ${COMET} > ${pred}/${testlog}.score

else
    echo "Argument a is not valid."
fi
