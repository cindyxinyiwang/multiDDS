DIR=/home/steven/Documents/GITHUB/multiDDS
source $DIR/venv/bin/activate

fairseq-preprocess --source-lang src --target-lang trg \
  --trainpref train.debug --validpref valid.debug --testpref test.debug \
  --joined-dictionary \
  --bpe sentencepiece \
  --destdir $DIR/data-bin