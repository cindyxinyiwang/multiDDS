

DATA_DIR=data/debug/

python preprocess.py -s src -t trg \
        --trainpref $DATA_DIR/train.debug \
        --joined-dictionary \
        --validpref $DATA_DIR/valid.debug \
        --testpref $DATA_DIR/test.debug \
        --workers 1 \
        --thresholdsrc 0 \
        --thresholdtgt 0 \
        --destdir data-bin/debug
	#--dataset-impl 'raw' \
	#--sde \
