LANS=(
  turtoaze
  turtoazecombine)

DATA_DIR=/projects/tir1/users/xinyiw1/fairseq/data/untokenized/
DATA_BIN=/projects/tir1/users/xinyiw1/fairseq/data-bin/ted_8_sepv/

#mkdir -p $DATA_BIN
#for i in ${!LANS[*]}; do
#  LAN=${LANS[$i]}
#  cat $DATA_DIR/"$LAN"_eng/ted-train.orig.spm8000."$LAN" >> $DATA_BIN/combined-train.spm8000.src
#  cat $DATA_DIR/"$LAN"_eng/ted-train.orig.spm8000.eng >> $DATA_BIN/combined-train.spm8000.eng
#
#done
#
#python preprocess.py -s src -t eng \
#  --trainpref $DATA_BIN/combined-train.spm8000 \
#  --joined-dictionary \
#  --workers 8 \
#  --thresholdsrc 0 \
#  --thresholdtgt 0 \
#  --destdir $DATA_BIN

for i in ${!LANS[*]}; do
  LAN=${LANS[$i]}
  python preprocess.py -s $LAN -t eng \
    --trainpref $DATA_DIR/"$LAN"_eng/ted-train.orig.spm8000 \
    --validpref $DATA_DIR/"$LAN"_eng/ted-dev.orig.spm8000 \
    --testpref $DATA_DIR/"$LAN"_eng/ted-test.orig.spm8000 \
    --srcdict $DATA_BIN/dict.src.txt \
    --tgtdict $DATA_BIN/dict.eng.txt \
    --workers 8 \
    --thresholdsrc 0 \
    --thresholdtgt 0 \
    --destdir $DATA_BIN

  python preprocess.py -s eng  -t $LAN \
    --trainpref $DATA_DIR/"$LAN"_eng/ted-train.orig.spm8000 \
    --validpref $DATA_DIR/"$LAN"_eng/ted-dev.orig.spm8000 \
    --testpref $DATA_DIR/"$LAN"_eng/ted-test.orig.spm8000 \
    --tgtdict $DATA_BIN/dict.src.txt \
    --srcdict $DATA_BIN/dict.eng.txt \
    --workers 8 \
    --thresholdsrc 0 \
    --thresholdtgt 0 \
    --destdir $DATA_BIN

done


