#!/bin/bash

#SBATCH --output=/project/jonmay_231/linghaoj/reproduce/slurm/debug-de.out
#SBATCH --error=/project/jonmay_231/linghaoj/reproduce/slurm/debug-de.err
#SBATCH --job-name=mega-de
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=isi
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
#SBATCH --open-mode=append
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=linghaojin@gmail.com

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

# Model
if [ -n "$a" ]; then a=$a ; else a=concat ; fi
if [ -n "$t" ]; then t=$t ; else t=reg ; fi
if [ -n "$s" ]; then s=$s ; else s=nonsf ; fi
if [ -n "$dropout" ]; then dropout=$dropout ; else dropout=0.2; fi
if [ -n "$activation_dropout" ]; then activation_dropout=$activation_dropout ; else activation_dropout=0.0 ; fi
if [ -n "$attention_dropout" ]; then attention_dropout=$attention_dropout ; else attention_dropout=0.0 ; fi
if [ -n "$model_size" ]; then model_size=$model_size ; else model_size=base ; fi
if [ -n "$num_workers" ]; then num_workers=$num_workers ; else num_workers=0 ; fi
if [ -n "$update_freq" ]; then update_freq=$update_freq ; else update_freq=16 ; fi
if [ -n "$max_tokens" ]; then max_tokens=$max_tokens ; else max_tokens=4096 ; fi

# Loss
if [ -n "$criterion" ]; then criterion=$criterion; else criterion=label_smoothed_cross_entropy; fi
if [ -n "$label_smoothing" ]; then label_smoothing=$label_smoothing; else label_smoothing=0.1; fi
if [ -n "$weight_decay" ]; then weight_decay=$weight_decay; else weight_decay=0.01; fi
if [ -n "$clip_norm" ]; then clip_norm=$clip_norm; else clip_norm=0.1; fi

# Optimization
if [ -n "$lr_scheduler" ]; then lr_scheduler=$lr_scheduler; else lr_scheduler=inverse_sqrt; fi
if [ -n "$lr" ]; then lr=$lr; else lr=1e-03; fi
if [ -n "$total_num_update" ]; then total_num_update=$total_num_update; else total_num_update=200000; fi 
if [ -n "$warmup_updates" ]; then warmup_updates=$$warmup_updates ; else warmup_updates=4000; fi
# if [ -n "$min_lr" ]; then min_lr=$min_lr ; else min_lr=1e-09 ; fi # stop training when the lr reaches this minimum (default -1.0)
# if [ -n "$warmup_init_lr" ]; then warmup_init_lr=$warmup_init_lr ; else warmup_init_lr=1e-07 ; fi
# if [ -n "$end_learning_rate" ]; then end_learning_rate=$end_learning_rate ; else end_learning_rate=1e-09 ; fi

if [ -n "$seed" ]; then seed=$seed ; else seed=42 ; fi
if [ -n "$wandb_project" ]; then wandb_project=$wandb_project ; else wandb_project=test ; fi
if [ -n "$src" ]; then src=$src ; else src=de ; fi
if [ -n "$tgt" ]; then tgt=$tgt ; else tgt=en ; fi

#########################################################################################################################

if [ $a = "mega" ]
then
    if [ $model_size = "base" ] 
    then
        arch="concat_mega"
    elif [ $model_size = "small" ] 
    then
        arch="concat_mega_iwslt"
    else
        arch="concat_mega_big"
    fi

    if [ $t = "src3" ]
    then
        srun --label fairseq-train ${bin} \
            --user-dir ${repo} --save-dir ${ckpt} \
            --task document_translation -s ${src} -t ${tgt} \
            --arch $arch --encoder-layers 6 --decoder-layers 6 --share-decoder-input-output-embed \
            --activation-fn silu --attention-activation-fn softmax \
            --encoder-n-dim 16 --encoder-chunk-size -1 \
            --normalization-type layernorm \
            --ddp-backend c10d \
            --num-workers $num_workers \
            --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm $clip_norm \
            --lr $lr --lr-scheduler $lr_scheduler  --warmup-updates $warmup_updates \
            --dropout $dropout --attention-dropout $attention_dropout --hidden-dropout 0.0 --activation-dropout $activation_dropout \
            --weight-decay $weight_decay --criterion $criterion --label-smoothing $label_smoothing \
            --max-tokens $max_tokens --update-freq $update_freq --seed $seed --max-update $total_num_update \
            --eval-bleu  --eval-bleu-remove-bpe sentencepiece \
            --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
            --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
            --keep-last-epochs 5 \
            --fp16 \
            --next-sent-ctx \
            --wandb-project $wandb_project

    elif [ $t = "reg" ]
    then 
        srun --label fairseq-train ${bin} \
            --user-dir ${repo} --save-dir ${ckpt} \
            --task document_translation -s ${src} -t ${tgt} \
            --source-context-size $n --target-context-size $m \
            --arch $arch --encoder-layers 6 --decoder-layers 6 --share-decoder-input-output-embed \
            --activation-fn silu --attention-activation-fn softmax \
            --encoder-n-dim 16 --encoder-chunk-size -1 \
            --normalization-type layernorm \
            --num-workers $num_workers \
            --ddp-backend c10d \
            --optimizer adam --adam-betas '(0.9, 0.98)'  --clip-norm $clip_norm \
            --lr $lr --lr-scheduler $lr_scheduler  --warmup-updates $warmup_updates \
            --dropout $dropout --attention-dropout $attention_dropout --hidden-dropout 0.0 --activation-dropout $activation_dropout \
            --weight-decay $weight_decay --criterion $criterion --label-smoothing $label_smoothing \
            --max-tokens $max_tokens --update-freq $update_freq --seed $seed --max-update $total_num_update \
            --eval-bleu  --eval-bleu-remove-bpe sentencepiece \
            --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
            --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
            --keep-last-epochs 5 \
            --fp16 \
            --shuffle_sample \
            --wandb-project $wandb_project
    fi

###############################################################################
elif [ $a = "concat" ]
then
    if [ $model_size = "base" ] 
    then
        arch=contextual_transformer
    elif [ $model_size = "small" ] 
    then
        arch=contextual_transformer_iwslt
    else
        arch=contextual_transformer_big
    fi

    if [ $t = "src3" ]
    then
        srun --label fairseq-train ${bin} \
            --user-dir ${repo} --save-dir ${ckpt} \
            --task document_translation -s ${src} -t ${tgt} \
            --arch $arch --share-decoder-input-output-embed \
            --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm $clip_norm \
            --lr $lr --lr-scheduler $lr_scheduler  --warmup-updates $warmup_updates \
            --weight-decay $weight_decay --criterion $criterion --label-smoothing $label_smoothing \
            --dropout $dropout \
            --num-workers $num_workers \
            --max-tokens $max_tokens --update-freq $update_freq --seed $seed --max-update $total_num_update \
            --eval-bleu \
            --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
            --eval-bleu-remove-bpe sentencepiece \
            --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
            --keep-last-epochs 5 \
            --fp16 \
            --next-sent-ctx \
            --wandb-project $wandb_project

    elif [ $t = "reg" ]
    then
        
        srun --label fairseq-train \
            ${bin} --user-dir ${repo} --save-dir ${ckpt} \
            --task document_translation -s ${src} -t ${tgt}\
            --arch $arch --share-decoder-input-output-embed \
            --source-context-size $n --target-context-size $m \
            --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm $clip_norm \
            --lr $lr --lr-scheduler $lr_scheduler  --warmup-updates $warmup_updates \
            --weight-decay $weight_decay --criterion $criterion --label-smoothing $label_smoothing \
            --dropout $dropout --coword-dropout 0 \
            --num-workers $num_workers \
            --max-tokens $max_tokens --update-freq $update_freq --seed $seed --max-update $total_num_update \
            --eval-bleu \
            --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
            --eval-bleu-remove-bpe sentencepiece \
            --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
            --keep-last-epochs 5 \
            --fp16 \
            --wandb-project $wandb_project
    fi

###############################################################################
elif [ $a = "xfmr" ]
then
    if [ $model_size = "base" ] 
    then
        arch="transformer"
    elif [ $model_size = "small" ] 
    then
        arch="transformer_iwslt_de_en"
    else
        arch="transformer_vaswani_wmt_en_de_big"
    fi

    # bsz ~1200
    srun --label fairseq-train ${bin} \
    --save-dir ${ckpt} \
	--task translation -s ${src} -t ${tgt} \
    --arch $arch --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm $clip_norm \
    --lr $lr --lr-scheduler $lr_scheduler  --warmup-updates $warmup_updates \
    --criterion $criterion --label-smoothing $label_smoothing --dropout $dropout --weight-decay $weight_decay \
    --max-tokens $max_tokens --update-freq $update_freq --seed $seed --max-update $total_num_update \
    --num-workers $num_workers \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe sentencepiece \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --keep-last-epochs 5 \
    --fp16 \
    --wandb-project $wandb_project

###############################################################################
else
    echo "Argument a is not valid."
fi
