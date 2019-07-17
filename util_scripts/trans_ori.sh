#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=dev
#SBATCH --time=2330
#SBATCH --mem=15GB
##SBATCH --output=checkpoints/ori_alignv1_spm8000_lstm/train-%j.out
##SBATCH --error=checkpoints/ori_alignv1_spm8000_lstm/train-%j.error


IL=ori
DATAPATH=data-bin/ori_alignv1_spm8000_jd/
MODEL=checkpoints/ori_alignv1_spm8000_lstm/checkpoint_best.pt
MODELNAME=lstm_spm_alignv1

mkdir -p trans_results/"$IL"/

python interactive.py $DATAPATH \
       	--path $MODEL \
	--batch-size 8 \
	--buffer-size 10 \
	--lenpen 1.2 \
	--remove-bpe sentencepiece \
	--beam 5  < test_data/ori_eng/set0-manual-test.spm8000.dnt0.ori  > ./trans_results/"$IL"/"$MODELNAME".set0-manual-test.dnt0.log

grep ^H ./trans_results/"$IL"/"$MODELNAME".set0-manual-test.dnt0.log | cut -f3 > ./trans_results/"$IL"/"$MODELNAME".set0-manual-test.dnt0.decode


python interactive.py $DATAPATH \
       	--path $MODEL \
	--batch-size 8 \
	--buffer-size 10 \
	--lenpen 1.2 \
	--remove-bpe sentencepiece \
	--beam 5  < test_data/ori_eng/setE-mono-standard.spm8000.dnt0.ori > ./trans_results/"$IL"/"$MODELNAME".setE.dnt0.log

grep ^H ./trans_results/"$IL"/"$MODELNAME".setE.dnt0.log | cut -f3 > ./trans_results/"$IL"/"$MODELNAME".setE.dnt0.decode

python interactive.py $DATAPATH \
       	--path $MODEL \
	--batch-size 8 \
	--buffer-size 10 \
	--lenpen 1.2 \
	--remove-bpe sentencepiece \
	--beam 5  < test_data/ori_eng/set0-test.orig.spm8000.dnt.align.ori > ./trans_results/"$IL"/"$MODELNAME".set0-test.dnt0.log

grep ^H ./trans_results/"$IL"/"$MODELNAME".set0-test.dnt0.log | cut -f3 > ./trans_results/"$IL"/"$MODELNAME".set0-test.dnt0.decode


