#!/bin/bash

#SBATCH --output=/project/jonmay_231/linghaoj/canmt-challenges/slurm/mbart-wmt17-p2p.out
#SBATCH --error=/project/jonmay_231/linghaoj/canmt-challenges/slurm/mbart-wmt17-p2p.err
#SBATCH --job-name=mbart-wmt17-p2p
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2
#SBATCH --partition=isi
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
#SBATCH --open-mode=truncate
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=linghaojin@gmail.com

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


src=zh_CN
tgt=en_XX

rm -rf $ckpt
mkdir -p $ckpt

langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

fairseq-train ${bin} --save-dir ${ckpt} \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbart_large --layernorm-embedding --share-decoder-input-output-embed \
  --task translation_from_pretrained_bart \
  --source-lang $src --target-lang $tgt \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 1e-04 --warmup-updates 2500 --max-update 40000 \
  --dropout 0.2 --attention-dropout 0.1 --weight-decay 0.01 \
  --max-tokens 1024 --update-freq 4 \
  --keep-last-epochs 5 \
  --seed ${seed} --log-format simple \
  --restore-file ${model}/checkpoint_best.pt \
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --langs ${langs} \
  --ddp-backend no_c10d \
  --fp16 \
  --wandb-project ${wandb_project}