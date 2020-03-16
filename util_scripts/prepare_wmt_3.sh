#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32



DATA_DIR=/projects/tir3/users/xinyiw1/wmt18/data/
DATA_BIN=/projects/tir1/users/xinyiw1/fairseq/data-bin/wmt4_sepv/

mkdir -p $DATA_BIN

#ILS=(
#  aze
#  tur
#  bel
#  rus
#  glg
#  por
#  slk
#  ces)

ILS=(
  de
  et
  fi
  tr)

#for LAN in `cat util_scripts/ted-langs-15.txt`; do
for i in ${!ILS[*]}; do
  LAN=${ILS[$i]}
  cat $DATA_DIR/"$LAN"-en/corpus.spm8000."$LAN" >> $DATA_BIN/combined-train.spm8000.sr
  cat $DATA_DIR/"$LAN"-en/corpus.spm8000.en >> $DATA_BIN/combined-train.spm8000.en
done

python preprocess.py -s sr -t en \
  --trainpref $DATA_BIN/combined-train.spm8000 \
  --workers 32 \
  --thresholdsrc 3 \
  --thresholdtgt 5 \
  --destdir $DATA_BIN
  #--joined-dictionary \

#for LAN in `cat util_scripts/ted-langs-15.txt`; do
for i in ${!ILS[*]}; do
  LAN=${ILS[$i]}
  python preprocess.py -s $LAN -t en \
    --trainpref $DATA_DIR/"$LAN"-en/corpus.spm8000 \
    --validpref $DATA_DIR/"$LAN"-en/newsdev2018.tc.spm8000 \
    --testpref $DATA_DIR/"$LAN"-en/newstest2018-"$LAN"en.spm8000 \
    --srcdict $DATA_BIN/dict.sr.txt \
    --tgtdict $DATA_BIN/dict.en.txt \
    --workers 32 \
    --thresholdsrc 0 \
    --thresholdtgt 0 \
    --destdir $DATA_BIN

  python preprocess.py -s en  -t $LAN \
    --trainpref $DATA_DIR/"$LAN"-en/corpus.spm8000 \
    --validpref $DATA_DIR/"$LAN"-en/newsdev2018.tc.spm8000 \
    --testpref $DATA_DIR/"$LAN"-en/newstest2018-en"$LAN".spm8000 \
    --tgtdict $DATA_BIN/dict.sr.txt \
    --srcdict $DATA_BIN/dict.en.txt \
    --workers 32 \
    --thresholdsrc 0 \
    --thresholdtgt 0 \
    --destdir $DATA_BIN
done


