LANS=(
  slk
  ces)

DATA_DIR=/home/xinyiw/data/ted_moses/
DATA_BIN=/home/xinyiw/fairseq/data-bin/ted_slkces/

#mkdir -p $DATA_BIN
#for i in ${!LANS[*]}; do
#  LAN=${LANS[$i]}
#  cat $DATA_DIR/"$LAN"_eng/ted-train.mtok.spm8000."$LAN" >> $DATA_BIN/combined-train.spm8000.src
#  cat $DATA_DIR/"$LAN"_eng/ted-train.mtok.spm8000.eng >> $DATA_BIN/combined-train.spm8000.eng
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
    --trainpref $DATA_DIR/"$LAN"_eng/ted-train.mtok.spm8000 \
    --validpref $DATA_DIR/"$LAN"_eng/ted-dev.mtok.spm8000 \
    --testpref $DATA_DIR/"$LAN"_eng/ted-test.mtok.spm8000 \
    --srcdict $DATA_BIN/dict.src.txt \
    --tgtdict $DATA_BIN/dict.eng.txt \
    --workers 8 \
    --thresholdsrc 0 \
    --thresholdtgt 0 \
    --destdir $DATA_BIN

  python preprocess.py -s eng  -t $LAN \
    --trainpref $DATA_DIR/"$LAN"_eng/ted-train.mtok.spm8000 \
    --validpref $DATA_DIR/"$LAN"_eng/ted-dev.mtok.spm8000 \
    --testpref $DATA_DIR/"$LAN"_eng/ted-test.mtok.spm8000 \
    --tgtdict $DATA_BIN/dict.src.txt \
    --srcdict $DATA_BIN/dict.eng.txt \
    --workers 8 \
    --thresholdsrc 0 \
    --thresholdtgt 0 \
    --destdir $DATA_BIN

done


