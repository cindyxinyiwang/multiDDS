#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=dev
#SBATCH --time=2330
#SBATCH --mem=15GB
##SBATCH --output=checkpoints/ilo_alignv1_spm8000_lstm/train-%j.out
##SBATCH --error=checkpoints/ilo_alignv1_spm8000_lstm/train-%j.error


IL=ilo
DATAPATH=data-bin/ilo_alignv1_spm8000_jd/
MODEL=checkpoints/ilo_alignv1_spm8000_lstm/checkpoint_best.pt
MODELNAME=lstm_spm_alignv1

mkdir -p trans_results/"$IL"/

python interactive.py $DATAPATH \
       	--path $MODEL \
	--batch-size 8 \
	--buffer-size 10 \
	--lenpen 1.2 \
	--remove-bpe sentencepiece \
	--beam 5  < test_data/ilo_eng/set0-manual-test.spm8000.dnt0.ilo  > ./trans_results/"$IL"/"$MODELNAME".set0-manual-test.dnt0.log

grep ^H ./trans_results/"$IL"/"$MODELNAME".set0-manual-test.dnt0.log | cut -f3 > ./trans_results/"$IL"/"$MODELNAME".set0-manual-test.dnt0.decode


python interactive.py $DATAPATH \
       	--path $MODEL \
	--batch-size 8 \
	--buffer-size 10 \
	--lenpen 1.2 \
	--remove-bpe sentencepiece \
	--beam 5  < test_data/ilo_eng/setE-mono-standard.spm8000.dnt0.ilo > ./trans_results/"$IL"/"$MODELNAME".setE.dnt0.log

grep ^H ./trans_results/"$IL"/"$MODELNAME".setE.dnt0.log | cut -f3 > ./trans_results/"$IL"/"$MODELNAME".setE.dnt0.decode

python interactive.py $DATAPATH \
       	--path $MODEL \
	--batch-size 8 \
	--buffer-size 10 \
	--lenpen 1.2 \
	--remove-bpe sentencepiece \
	--beam 5  < test_data/ilo_eng/set0-test.ilog.spm8000.dnt.align.ilo > ./trans_results/"$IL"/"$MODELNAME".set0-test.dnt0.log

grep ^H ./trans_results/"$IL"/"$MODELNAME".set0-test.dnt0.log | cut -f3 > ./trans_results/"$IL"/"$MODELNAME".set0-test.dnt0.decode


