

DATA_DIR=data/bt_iwslt15_ende/
DATA_BIN=data-bin/bt_iwslt15_ende10k/
#
python preprocess.py -s en -t de \
        --trainpref $DATA_DIR/train10k \
        --joined-dictionary \
        --validpref $DATA_DIR/tst2012 \
        --testpref $DATA_DIR/tst2013 \
        --workers 1 \
        --thresholdsrc 0 \
        --thresholdtgt 0 \
        --destdir $DATA_BIN

python preprocess.py -s de -t en \
        --trainpref $DATA_DIR/train10k \
        --tgtdict $DATA_BIN/dict.en.txt \
        --srcdict $DATA_BIN/dict.de.txt \
        --validpref $DATA_DIR/tst2012 \
        --testpref $DATA_DIR/tst2013 \
        --workers 1 \
        --thresholdsrc 0 \
        --thresholdtgt 0 \
        --destdir data-bin/bt_iwslt15_ende
python preprocess.py -s de -t None \
        --trainpref $DATA_DIR/mono10k/train \
        --tgtdict $DATA_BIN/dict.en.txt \
        --srcdict $DATA_BIN/dict.de.txt \
	--only-source \
        --workers 1 \
        --thresholdsrc 0 \
        --thresholdtgt 0 \
        --destdir $DATA_BIN




#python preprocess.py -s en -t de \
#        --trainpref $DATA_DIR/train.combined \
#        --joined-dictionary \
#        --validpref $DATA_DIR/tst2012 \
#        --testpref $DATA_DIR/tst2013 \
#        --workers 1 \
#        --thresholdsrc 0 \
#        --thresholdtgt 0 \
#        --destdir data-bin/iwslt15_ende
#	#--dataset-impl 'raw' \
#	#--sde \
#

#python preprocess.py -s tedenavesort50k -t teddeavesort50k \
#        --trainpref $DATA_DIR/train \
#        --srcdict $DATA_BIN/dict.en.txt \
#        --tgtdict $DATA_BIN/dict.de.txt \
#        --validpref $DATA_DIR/tst2012 \
#        --testpref $DATA_DIR/tst2013 \
#        --workers 1 \
#        --thresholdsrc 0 \
#        --thresholdtgt 0 \
#        --destdir data-bin/iwslt15_ende
#        #--dataset-impl "raw" \

#python preprocess.py -s tedenraw1o210ks1 -t tedderaw1o210ks1 \
#python preprocess.py -s teden1o2sort5k -t tedde1o2sort5k \
#        --trainpref $DATA_DIR/train \
#        --srcdict $DATA_BIN/dict.en.txt \
#        --tgtdict $DATA_BIN/dict.de.txt \
#        --validpref $DATA_DIR/tst2012 \
#        --testpref $DATA_DIR/tst2013 \
#        --workers 1 \
#        --thresholdsrc 0 \
#        --thresholdtgt 0 \
#        --dataset-impl "raw" \
#        --destdir data-bin/iwslt15_ende

#python preprocess.py -s wmten500K -t wmtde500K \
#        --trainpref $DATA_DIR/train \
#        --srcdict $DATA_BIN/dict.en.txt \
#        --tgtdict $DATA_BIN/dict.de.txt \
#        --validpref $DATA_DIR/tst2012 \
#        --testpref $DATA_DIR/tst2013 \
#        --workers 1 \
#        --thresholdsrc 0 \
#        --thresholdtgt 0 \
#        --destdir data-bin/iwslt15_ende
#


