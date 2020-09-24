#!/bin/bash


MODEL_DIR=/home/steven/Documents/GITHUB/multiDDS
source $MODEL_DIR/venv/bin/activate

fairseq-generate data-bin/ted_8_related/ \
          --task multilingual_translation \
          --gen-subset test \
          --path $MODEL_DIR/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	        --sacrebleu \
          --lang-pairs "aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng" \
          --source-lang aze --target-lang eng \
          --beam 5

python generate.py data-bin/ted_8_related/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	  --sacrebleu \
          --lang-pairs "aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng" \
          --source-lang tur --target-lang eng \
          --beam 5

python generate.py data-bin/ted_8_related/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	  --sacrebleu \
          --lang-pairs "aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng" \
          --source-lang bel --target-lang eng \
          --beam 5

python generate.py data-bin/ted_8_related/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	  --sacrebleu \
          --lang-pairs "aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng" \
          --source-lang rus --target-lang eng \
          --beam 5

python generate.py data-bin/ted_8_related/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	  --sacrebleu \
          --lang-pairs "aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng" \
          --source-lang glg --target-lang eng \
          --beam 5

python generate.py data-bin/ted_8_related/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	  --sacrebleu \
          --lang-pairs "aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng" \
          --source-lang por --target-lang eng \
          --beam 5

python generate.py data-bin/ted_8_related/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	  --sacrebleu \
          --lang-pairs "aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng" \
          --source-lang slk --target-lang eng \
          --beam 5

python generate.py data-bin/ted_8_related/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	  --sacrebleu \
          --lang-pairs "aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng" \
          --source-lang ces --target-lang eng \
          --beam 5

