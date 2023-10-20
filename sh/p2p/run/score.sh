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

mkdir $RESFILE

grep ^S $RESFILE/$testlog.txt | sed 's/^S-//g' |  cut -f2- | sacremoses detokenize > $RESFILE/$testlog.src.detok
grep ^H $RESFILE/$testlog.txt | sed 's/^H-//g' |  cut -f3- | sacremoses detokenize > $RESFILE/$testlog.pred.detok
grep ^T $RESFILE/$testlog.txt | sed 's/^T-//g' |  cut -f2- | sacremoses detokenize > $RESFILE/$testlog.ref.detok

sacrebleu $RESFILE/$testlog.ref.detok < $RESFILE/$testlog.pred.detok > $RESFILE/$testlog.score.detok
blonde -r $RESFILE/$testlog.ref.detok -s $RESFILE/$testlog.pred.detok >> $RESFILE/$testlog.score.detok
comet-score -s $RESFILE/$testlog.src.detok -t $RESFILE/$testlog.pred.detok -r $RESFILE/$testlog.ref.detok >> $RESFILE/$testlog.score.detok # reference-based