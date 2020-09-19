DIR=/home/steven/Documents/GITHUB/multiDDS
source $DIR/venv/bin/activate


fairseq-generate $DIR/data-bin/  \
          --task multilingual_translation \
          --gen-subset test \
          --path $DIR/checkpoints/debug/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	        --sacrebleu \
          --lang-pairs "src-trg" \
          --source-lang src --target-lang trg \
          --beam 5

