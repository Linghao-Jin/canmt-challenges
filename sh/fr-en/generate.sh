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
if [ -n "$script_path" ]; then script_path=$script_path ; else script_path=$root/scripts ; fi
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

if [ "$a" = "xfmr" ] || [ "$a" = "mega" ]; 
then
    fairseq-generate $bin \
        --user-dir ${model} \
        --path ${ckpt}/${checkpoint} \
        --batch-size ${batch_size} \
        --beam ${beam} \
        --remove-bpe ${remove_bpe} \
        --scoring sacrebleu \
        > ${pred}/${testlog}.log

    grep ^T ${pred}/${testlog}.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- | sacremoses detokenize > ${pred}/${testlog}.ref.detok
    grep ^H ${pred}/${testlog}.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- | sacremoses detokenize > ${pred}/${testlog}.sys.detok
    grep ^S ${pred}/${testlog}.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- | sacremoses detokenize > ${pred}/${testlog}.src.detok

    sacrebleu ${pred}/$testlog.ref.detok < ${pred}/$testlog.sys.detok > ${pred}/${testlog}.score.detok
    blonde -r ${pred}/$testlog.ref.detok -s ${pred}/$testlog.sys.detok >> ${pred}/${testlog}.score.detok
    comet-score -s ${pred}/$testlog.src.detok -t ${pred}/$testlog.sys.detok -r ${pred}/$testlog.ref.detok >> ${pred}/${testlog}.score.detok # reference-based

elif [[ $a = "concat" ]]
then
    if [[ $t = "src3" ]]
    then
        python ${repo}/docmt_translate.py \
        --path ${ckpt} \
        --checkpoint_file ${checkpoint} \
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
        --checkpoint_file ${checkpoint} \
        --source-lang ${src} --target-lang ${tgt} \
        --source-file $source_file \
        --predictions-file ${pred}/${testlog}.sys \
        --docids-file $docids_file \
        --beam $beam
    
    fi

    cat ${pred}/${testlog}.sys | sacremoses detokenize > ${pred}/${testlog}.sys.detok
    cat $source_file | sacremoses detokenize > ${pred}/${testlog}.src.detok
    cat $target_file | sacremoses detokenize > ${pred}/${testlog}.ref.detok
    
    sacrebleu ${pred}/$testlog.ref.detok < ${pred}/$testlog.sys.detok > ${pred}/${testlog}.score.detok
    blonde -r ${pred}/$testlog.ref.detok -s ${pred}/$testlog.sys.detok >> ${pred}/${testlog}.score.detok
    comet-score -s ${pred}/$testlog.src.detok -t ${pred}/$testlog.sys.detok -r ${pred}/$testlog.ref.detok >> ${pred}/${testlog}.score.detok # reference-based


elif [[ $a = "concat-mega" ]]
then
    if [[ $t = "src3" ]]
    then
        python ${repo}/docmt_translate.py \
        --path ${ckpt} \
        --checkpoint_file ${checkpoint} \
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
        --checkpoint_file ${checkpoint} \
        --source-lang ${src} --target-lang ${tgt} \
        --source-file $source_file \
        --predictions-file ${pred}/${testlog}.sys \
        --docids-file $docids_file \
        --beam $beam
    
    fi

    cat ${pred}/${testlog}.sys | sacremoses detokenize > ${pred}/${testlog}.sys.detok
    cat $source_file | sacremoses detokenize > ${pred}/${testlog}.src.detok
    cat $target_file | sacremoses detokenize > ${pred}/${testlog}.ref.detok
    
    sacrebleu ${pred}/$testlog.ref.detok < ${pred}/$testlog.sys.detok > ${pred}/${testlog}.score.detok
    blonde -r ${pred}/$testlog.ref.detok -s ${pred}/$testlog.sys.detok >> ${pred}/${testlog}.score.detok
    comet-score -s ${pred}/$testlog.src.detok -t ${pred}/$testlog.sys.detok -r ${pred}/$testlog.ref.detok >> ${pred}/${testlog}.score.detok # reference-based

else
    echo "Argument a is not valid."
fi
