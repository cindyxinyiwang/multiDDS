
# you need to first download the raw data from https://drive.google.com/file/d/1Va9KHGjPNd9pBvfzujq7aPuGrY-wRn91/view?usp=sharing
# unzip the file and put it in the working directory

DATA_DIR=ted/
DATA_BIN=data-bin/ted_8_related/
vocab_size=8000

# process related language group. Change language names if you want to process other data.
LANS=(
  aze
  tur
  bel
  rus
  glg
  por
  slk
  ces)

# sentencepiece the data
for i in ${!LANS[*]}; do
  LAN=${LANS[$i]}
  mkdir -p "$DATA_DIR"/eng

  python train-spm.py \
    --input="$DATA_DIR"/"$LAN"_eng/"$DATA_PRE"-train.orig."$LAN" \
    --model_prefix="$DATA_DIR"/"$LAN"_eng/spm"$vocab_size.orig.$LAN" \
    --vocab_size="$vocab_size"

  for f in "$DATA_DIR"/"$LAN"_eng/*.orig."$LAN"; 
  do
    python $UDIR/run-spm.py \
      --model="$DATA_DIR"/"$LAN"_eng/spm"$vocab_size.orig.$LAN".model \
      < $f \
      > ${f/orig/orig.spm$vocab_size} 
  done

  cat "$DATA_DIR"/"$LAN"_eng/ted-train.orig.eng >> "$DATA_BIN"/eng/ted-train.orig.eng
done

python train-spm.py \
  --input="$DATA_DIR"/eng/ted-train.orig.eng \
  --model_prefix="$DATA_DIR"/eng/spm"$vocab_size.orig.eng" \
  --vocab_size="$vocab_size"

for i in ${!LANS[*]}; do
  LAN=${LANS[$i]}

  for f in "$DATA_DIR"/"$LAN"_eng/*.orig.eng; 
  do
    python $UDIR/run-spm.py \
      --model="$DATA_DIR"/eng/spm"$vocab_size.orig.eng".model \
      < $f \
      > ${f/orig/orig.spm$vocab_size} 
  done
done

# binarize the data for fairseq
for i in ${!LANS[*]}; do
  LAN=${LANS[$i]}
  cat $DATA_DIR/"$LAN"_eng/ted-train.orig.spm"$vocab_size"."$LAN" >> $DATA_BIN/combined-train.spm"$vocab_size".src
  cat $DATA_DIR/"$LAN"_eng/ted-train.orig.spm"$vocab_size".eng >> $DATA_BIN/combined-train.spm"$vocab_size".eng
done

python preprocess.py -s src -t eng \
  --trainpref $DATA_BIN/combined-train.spm"$vocab_size" \
  --workers 8 \
  --thresholdsrc 0 \
  --thresholdtgt 0 \
  --destdir $DATA_BIN

for i in ${!LANS[*]}; do
  LAN=${LANS[$i]}
  python preprocess.py -s $LAN -t eng \
    --trainpref $DATA_DIR/"$LAN"_eng/ted-train.orig.spm"$vocab_size" \
    --validpref $DATA_DIR/"$LAN"_eng/ted-dev.orig.spm"$vocab_size" \
    --testpref $DATA_DIR/"$LAN"_eng/ted-test.orig.spm"$vocab_size" \
    --srcdict $DATA_BIN/dict.src.txt \
    --tgtdict $DATA_BIN/dict.eng.txt \
    --workers 8 \
    --thresholdsrc 0 \
    --thresholdtgt 0 \
    --destdir $DATA_BIN

  python preprocess.py -s eng  -t $LAN \
    --trainpref $DATA_DIR/"$LAN"_eng/ted-train.orig.spm"$vocab_size" \
    --validpref $DATA_DIR/"$LAN"_eng/ted-dev.orig.spm"$vocab_size" \
    --testpref $DATA_DIR/"$LAN"_eng/ted-test.orig.spm"$vocab_size" \
    --tgtdict $DATA_BIN/dict.src.txt \
    --srcdict $DATA_BIN/dict.eng.txt \
    --workers 8 \
    --thresholdsrc 0 \
    --thresholdtgt 0 \
    --destdir $DATA_BIN
done


