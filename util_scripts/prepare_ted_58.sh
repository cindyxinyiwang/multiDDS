#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0
#SBATCH --mem=20GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16



DATA_DIR=/projects/tir1/users/xinyiw1/fairseq/data/
DATA_BIN=/projects/tir1/users/xinyiw1/fairseq/data-bin/ted_eightdiverse_sepv/

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
  bos
  mar
  hin
  mkd
  ell
  bul
  fra
  kor)

#for LAN in `cat util_scripts/ted-langs-15.txt`; do
for i in ${!ILS[*]}; do
  LAN=${ILS[$i]}
  cat $DATA_DIR/"$LAN"_eng/ted-train.mtok.spm8000."$LAN" >> $DATA_BIN/combined-train.spm8000.src
  cat $DATA_DIR/"$LAN"_eng/ted-train.mtok.spm8000.eng >> $DATA_BIN/combined-train.spm8000.eng
done

python preprocess.py -s src -t eng \
  --trainpref $DATA_BIN/combined-train.spm8000 \
  --workers 16 \
  --thresholdsrc 0 \
  --thresholdtgt 0 \
  --destdir $DATA_BIN
  #--joined-dictionary \

#for LAN in `cat util_scripts/ted-langs-15.txt`; do
for i in ${!ILS[*]}; do
  LAN=${ILS[$i]}
  python preprocess.py -s $LAN -t eng \
    --trainpref $DATA_DIR/"$LAN"_eng/ted-train.mtok.spm8000 \
    --validpref $DATA_DIR/"$LAN"_eng/ted-dev.mtok.spm8000 \
    --testpref $DATA_DIR/"$LAN"_eng/ted-test.mtok.spm8000 \
    --srcdict $DATA_BIN/dict.src.txt \
    --tgtdict $DATA_BIN/dict.eng.txt \
    --workers 16 \
    --thresholdsrc 0 \
    --thresholdtgt 0 \
    --destdir $DATA_BIN

  python preprocess.py -s eng  -t $LAN \
    --trainpref $DATA_DIR/"$LAN"_eng/ted-train.mtok.spm8000 \
    --validpref $DATA_DIR/"$LAN"_eng/ted-dev.mtok.spm8000 \
    --testpref $DATA_DIR/"$LAN"_eng/ted-test.mtok.spm8000 \
    --tgtdict $DATA_BIN/dict.src.txt \
    --srcdict $DATA_BIN/dict.eng.txt \
    --workers 16 \
    --thresholdsrc 0 \
    --thresholdtgt 0 \
    --destdir $DATA_BIN
done


