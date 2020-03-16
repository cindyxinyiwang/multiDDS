#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32



DATA_DIR=/projects/tir3/users/xinyiw1/wmt18/data/
DATA_BIN=/projects/tir1/users/xinyiw1/fairseq/data-bin/wmt18_seg_et/

mkdir -p $DATA_BIN

ILS=(
  10aa
  10ab
  10ac
  10ad
  10ae
  10af
  10ag
  10ah
  10ai
  10aj)

LAN=et

#python preprocess.py -s et -t en \
#  --trainpref $DATA_DIR/"$LAN"-en/corpus.spm16000 \
#  --workers 32 \
#  --thresholdsrc 3 \
#  --thresholdtgt 5 \
#  --destdir $DATA_BIN
#  #--joined-dictionary \

#for LAN in `cat util_scripts/ted-langs-15.txt`; do
for i in ${!ILS[*]}; do
  SUF=${ILS[$i]}
  #python preprocess.py -s et.$SUF -t en.$SUF \
  #  --trainpref $DATA_DIR/"$LAN"-en/corpus.spm16000 \
  #  --validpref $DATA_DIR/"$LAN"-en/newsdev2018.tc.spm16000 \
  #  --testpref $DATA_DIR/"$LAN"-en/newstest2018-"$LAN"en.spm16000 \
  #  --srcdict $DATA_BIN/dict.et.txt \
  #  --tgtdict $DATA_BIN/dict.en.txt \
  #  --workers 32 \
  #  --thresholdsrc 0 \
  #  --thresholdtgt 0 \
  #  --destdir $DATA_BIN

  #python preprocess.py -s et.$SUF -t en.$SUF \
  #  --trainpref $DATA_DIR/"$LAN"-en/corpus.spm16000 \
  #  --validpref $DATA_DIR/"$LAN"-en/newsdev2018.tc.spm16000 \
  #  --testpref $DATA_DIR/"$LAN"-en/newstest2018-en"$LAN".spm16000 \
  #  --tgtdict $DATA_BIN/dict.et.txt \
  #  --srcdict $DATA_BIN/dict.en.txt \
  #  --workers 32 \
  #  --thresholdsrc 0 \
  #  --thresholdtgt 0 \
  #  --destdir $DATA_BIN
done

python preprocess.py -s et -t en \
  --trainpref $DATA_DIR/"$LAN"-en/corpus.spm16000 \
  --validpref $DATA_DIR/"$LAN"-en/newsdev2018.tc.spm16000 \
  --testpref $DATA_DIR/"$LAN"-en/newstest2018-"$LAN"en.spm16000 \
  --srcdict $DATA_BIN/dict.et.txt \
  --tgtdict $DATA_BIN/dict.en.txt \
  --workers 32 \
  --thresholdsrc 0 \
  --thresholdtgt 0 \
  --destdir $DATA_BIN


