#!/bin/bash
#SBATCH --output=/project/jonmay_231/linghaoj/canmt-challenges/slurm/generate_finetuned.out
#SBATCH --error=/project/jonmay_231/linghaoj/canmt-challenges/slurm/generate_finetuned.err
#SBATCH --job-name=bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=isi
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
#SBATCH --open-mode=append
#SBATCH --time=1-00:00:00

set -e

root=/project/jonmay_231/linghaoj/canmt-challenges
scripts=$root/scripts
ckpt_name=checkpoint_best.pt
testlog=best

cd $root

for seed in 22 42 73
do
    for book in samples samples-sf
    do
        ##################################################
        # scratch 

        src=zh
        tgt=en
        bin_name=bin-wmt17
        ckpt=$root/ckpt-p2p/$book/xfmr-scratch-p2p-$seed

        bin=$root/data/p2p/$book/$bin_name
        resfile=$ckpt/result
        mkdir -p $resfile

        fairseq-generate $bin \
            --path $ckpt/$ckpt_name \
            --task translation \
            --remove-bpe \
            --sacrebleu \
            --skip-invalid-size-inputs-valid-test \
            > $resfile/$testlog.txt

        bash sh/p2p/run/score.sh --RESFILE=$resfile --testlog=$testlog

        ##################################################
        # custom lightconv(wmt17)

        src=zh
        tgt=en
        bin_name=bin-custom-wmt17
        ckpt=$root/ckpt-p2p/$book/lightconv-wmt17-p2p-$seed

        bin=$root/data/p2p/$book/$bin_name
        resfile=$ckpt/result
        mkdir -p $resfile

        fairseq-generate $bin \
            --path $ckpt/$ckpt_name \
            --task translation \
            --remove-bpe \
            --sacrebleu \
            --skip-invalid-size-inputs-valid-test \
            > $resfile/$testlog.txt

        bash sh/p2p/run/score.sh --RESFILE=$resfile --testlog=$testlog

        ##################################################
        # xfmr-wmt17

        src=zh
        tgt=en
        bin_name=bin-custom-wmt17
        ckpt=$root/ckpt-p2p/$book/xfmr-wmt17-p2p-$seed

        bin=$root/data/p2p/$book/$bin_name
        resfile=$ckpt/result
        mkdir -p $resfile

        fairseq-generate $bin \
            --path $ckpt/$ckpt_name \
            --task translation \
            --remove-bpe \
            --sacrebleu \
            --skip-invalid-size-inputs-valid-test \
            > $resfile/$testlog.txt

        bash sh/p2p/run/score.sh --RESFILE=$resfile --testlog=$testlog

        ##################################################
        # xfmr-big-wmt17

        src=zh
        tgt=en
        bin_name=bin-custom-wmt17
        ckpt=$root/ckpt-p2p/$book/xfmr-big-wmt17-p2p-$seed

        bin=$root/data/p2p/$book/$bin_name
        resfile=$ckpt/result
        mkdir -p $resfile

        fairseq-generate $bin \
            --path $ckpt/$ckpt_name \
            --task translation \
            --remove-bpe \
            --sacrebleu \
            --skip-invalid-size-inputs-valid-test \
            > $resfile/$testlog.txt

        bash sh/p2p/run/score.sh --RESFILE=$resfile --testlog=$testlog

        ##################################################
        # mbart-wmt17

        src=zh_CN 
        tgt=en_XX 
        bin_name=bin-mbart
        ckpt=$root/ckpt-p2p/$book/mbart-wmt17-p2p-$seed
        langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

        bin=$root/data/p2p/$book/$bin_name
        resfile=$ckpt/result
        mkdir -p $resfile

        fairseq-generate $bin \
            --path=$ckpt/$ckpt_name \
            --task translation_from_pretrained_bart --gen-subset test -s $src -t $tgt \
            --remove-bpe 'sentencepiece' --langs $langs \
            --skip-invalid-size-inputs-valid-test \
            --sacrebleu \
            > $resfile/$testlog.txt

        bash sh/p2p/run/score.sh --RESFILE=$resfile --testlog=$testlog

    done
done