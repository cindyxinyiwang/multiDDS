

DATA_DIR=prep/
DATA_BIN=data-bin/bt_iwslt_deen30k/
#
python preprocess.py -s de -t en \
        --trainpref $DATA_DIR/train.de-en.30k.spm16000 \
        --joined-dictionary \
        --validpref $DATA_DIR/valid.de-en.spm16000 \
        --testpref $DATA_DIR/test.de-en.spm16000 \
        --workers 1 \
        --thresholdsrc 0 \
        --thresholdtgt 0 \
        --destdir $DATA_BIN

python preprocess.py -s en -t de \
        --trainpref $DATA_DIR/train.de-en.30k.spm16000 \
        --tgtdict $DATA_BIN/dict.en.txt \
        --srcdict $DATA_BIN/dict.en.txt \
        --validpref $DATA_DIR/valid.de-en.spm16000 \
        --testpref $DATA_DIR/test.de-en.spm16000 \
        --workers 1 \
        --thresholdsrc 0 \
        --thresholdtgt 0 \
        --destdir $DATA_BIN

python preprocess.py -s en -t None \
        --trainpref $DATA_DIR/mono30k/train.de-en.30k.spm16000 \
        --tgtdict $DATA_BIN/dict.en.txt \
        --srcdict $DATA_BIN/dict.de.txt \
	--only-source \
        --workers 1 \
        --thresholdsrc 0 \
        --thresholdtgt 0 \
        --destdir $DATA_BIN


