
DATA_DIR=data/untokenized/
DATA_BIN=data-bin/bt_glgpor/

LANS=(
  glg
  por)


mkdir -p $DATA_BIN
for i in ${!LANS[*]}; do
  LAN=${LANS[$i]}
  cat $DATA_DIR/"$LAN"_eng/ted-train.orig.spm8000."$LAN" >> $DATA_BIN/combined-train.spm8000.src
  cat $DATA_DIR/"$LAN"_eng/ted-train.orig.spm8000.eng >> $DATA_BIN/combined-train.spm8000.eng

done

python preprocess.py -s src -t eng \
  --trainpref $DATA_BIN/combined-train.spm8000 \
  --joined-dictionary \
  --workers 8 \
  --thresholdsrc 0 \
  --thresholdtgt 0 \
  --destdir $DATA_BIN


python preprocess.py -s glg -t eng \
        --trainpref $DATA_DIR/glg_eng/ted-train.orig.spm8000 \
        --validpref $DATA_DIR/glg_eng/ted-dev.orig.spm8000 \
        --testpref $DATA_DIR/glg_eng/ted-test.orig.spm8000 \
        --srcdict $DATA_BIN/dict.src.txt \
        --tgtdict $DATA_BIN/dict.eng.txt \
        --workers 1 \
        --thresholdsrc 0 \
        --thresholdtgt 0 \
        --destdir $DATA_BIN

python preprocess.py -s eng -t glg \
        --trainpref $DATA_DIR/glg_eng/ted-train.orig.spm8000 \
        --validpref $DATA_DIR/glg_eng/ted-dev.orig.spm8000 \
        --testpref $DATA_DIR/glg_eng/ted-test.orig.spm8000 \
        --srcdict $DATA_BIN/dict.src.txt \
        --tgtdict $DATA_BIN/dict.eng.txt \
        --workers 1 \
        --thresholdsrc 0 \
        --thresholdtgt 0 \
        --destdir $DATA_BIN
python preprocess.py -s por -t eng \
        --trainpref $DATA_DIR/por_eng/ted-train.orig.spm8000 \
        --validpref $DATA_DIR/por_eng/ted-dev.orig.spm8000 \
        --testpref $DATA_DIR/por_eng/ted-test.orig.spm8000 \
        --srcdict $DATA_BIN/dict.src.txt \
        --tgtdict $DATA_BIN/dict.eng.txt \
        --workers 1 \
        --thresholdsrc 0 \
        --thresholdtgt 0 \
        --destdir $DATA_BIN
python preprocess.py -s eng -t por \
        --trainpref $DATA_DIR/por_eng/ted-train.orig.spm8000 \
        --validpref $DATA_DIR/por_eng/ted-dev.orig.spm8000 \
        --testpref $DATA_DIR/por_eng/ted-test.orig.spm8000 \
        --srcdict $DATA_BIN/dict.src.txt \
        --tgtdict $DATA_BIN/dict.eng.txt \
        --workers 1 \
        --thresholdsrc 0 \
        --thresholdtgt 0 \
        --destdir $DATA_BIN


python preprocess.py -s eng -t None \
        --trainpref $DATA_DIR/por_eng/ted-train.orig.spm8000 \
        --tgtdict $DATA_BIN/dict.src.txt \
        --srcdict $DATA_BIN/dict.eng.txt \
	--only-source \
        --workers 1 \
        --thresholdsrc 0 \
        --thresholdtgt 0 \
        --destdir $DATA_BIN


