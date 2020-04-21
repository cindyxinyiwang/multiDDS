#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --mem=15GB

OUTDIR=$1

python generate.py data-bin/ted_8_diverse/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	  --encoder-langtok "tgt" \
          --lang-pairs "eng-bos,eng-mar,eng-hin,eng-mkd,eng-ell,eng-bul,eng-fra,eng-kor" \
          --source-lang eng --target-lang bos \
          --beam 5   > "$OUTDIR"/test_engbos.log

python generate.py data-bin/ted_8_diverse/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	  --encoder-langtok "tgt" \
          --lang-pairs "eng-bos,eng-mar,eng-hin,eng-mkd,eng-ell,eng-bul,eng-fra,eng-kor" \
          --source-lang eng --target-lang mar \
          --beam 5   > "$OUTDIR"/test_engmar.log

python generate.py data-bin/ted_8_diverse/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	  --encoder-langtok "tgt" \
          --lang-pairs "eng-bos,eng-mar,eng-hin,eng-mkd,eng-ell,eng-bul,eng-fra,eng-kor" \
          --source-lang eng --target-lang hin \
          --beam 5   > "$OUTDIR"/test_enghin.log

python generate.py data-bin/ted_8_diverse/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	  --encoder-langtok "tgt" \
          --lang-pairs "eng-bos,eng-mar,eng-hin,eng-mkd,eng-ell,eng-bul,eng-fra,eng-kor" \
          --source-lang eng --target-lang mkd \
          --beam 5   > "$OUTDIR"/test_engmkd.log

python generate.py data-bin/ted_8_diverse/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	  --encoder-langtok "tgt" \
          --lang-pairs "eng-bos,eng-mar,eng-hin,eng-mkd,eng-ell,eng-bul,eng-fra,eng-kor" \
          --source-lang eng --target-lang ell \
          --beam 5   > "$OUTDIR"/test_engell.log

python generate.py data-bin/ted_8_diverse/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	  --encoder-langtok "tgt" \
          --lang-pairs "eng-bos,eng-mar,eng-hin,eng-mkd,eng-ell,eng-bul,eng-fra,eng-kor" \
          --source-lang eng --target-lang bul \
          --beam 5   > "$OUTDIR"/test_engbul.log

python generate.py data-bin/ted_8_diverse/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	  --encoder-langtok "tgt" \
          --lang-pairs "eng-bos,eng-mar,eng-hin,eng-mkd,eng-ell,eng-bul,eng-fra,eng-kor" \
          --source-lang eng --target-lang fra \
          --beam 5   > "$OUTDIR"/test_engfra.log

python generate.py data-bin/ted_8_diverse/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	  --encoder-langtok "tgt" \
          --lang-pairs "eng-bos,eng-mar,eng-hin,eng-mkd,eng-ell,eng-bul,eng-fra,eng-kor" \
          --source-lang eng --target-lang kor \
          --beam 5   > "$OUTDIR"/test_engkor.log

