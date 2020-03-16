

DATA_DIR=data/dialects/
DATA_BIN=data-bin/dialects/
#
python preprocess.py -s fra -t eng \
        --trainpref $DATA_DIR/train.orig.spm8000 \
        --joined-dictionary \
        --workers 1 \
        --thresholdsrc 0 \
        --thresholdtgt 0 \
        --destdir $DATA_BIN

python preprocess.py -s fra -t eng \
        --trainpref $DATA_DIR/fra_eng/ted-train.orig.spm8000 \
        --srcdict $DATA_BIN/dict.fra.txt \
        --tgtdict $DATA_BIN/dict.eng.txt \
        --validpref $DATA_DIR/fra_eng/ted-dev.orig.spm8000 \
        --testpref $DATA_DIR/fra_eng/ted-test.orig.spm8000 \
        --workers 1 \
        --thresholdsrc 0 \
        --thresholdtgt 0 \
        --destdir $DATA_BIN

python preprocess.py -s XXfr-ca -t eng \
        --trainpref $DATA_DIR/XXfr-ca_eng/ted-train.orig.spm8000 \
        --srcdict $DATA_BIN/dict.fra.txt \
        --tgtdict $DATA_BIN/dict.eng.txt \
        --validpref $DATA_DIR/XXfr-ca_eng/ted-dev.orig.spm8000 \
        --testpref $DATA_DIR/XXfr-ca_eng/ted-test.orig.spm8000 \
        --workers 1 \
        --thresholdsrc 0 \
        --thresholdtgt 0 \
        --destdir $DATA_BIN


