#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --mem=15GB

##SBATCH --job-name=fw_slk-eng
##SBATCH --output=checkpoints/train_logs/fw_slk-eng_train-%j.out
##SBATCH --output=checkpoints/train_logs/fw_slk-eng_train-%j.err

#export PYTHONPATH="$(pwd)"
#export CUDA_VISIBLE_DEVICES="2"

OUTDIR=$1

python generate.py data-bin/ted_8_sepv/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --score-reference \
          --remove-bpe sentencepiece \
	  --encoder-langtok "tgt" \
          --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
          --source-lang eng --target-lang aze \
          --beam 5   > "$OUTDIR"/test_engaze.log

python generate.py data-bin/ted_8_sepv/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --score-reference \
          --remove-bpe sentencepiece \
	  --encoder-langtok "tgt" \
          --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
          --source-lang eng --target-lang tur \
          --beam 5   > "$OUTDIR"/test_engtur.log

python generate.py data-bin/ted_8_sepv/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --score-reference \
          --remove-bpe sentencepiece \
	  --encoder-langtok "tgt" \
          --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
          --source-lang eng --target-lang bel \
          --beam 5   > "$OUTDIR"/test_engbel.log

python generate.py data-bin/ted_8_sepv/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --score-reference \
          --remove-bpe sentencepiece \
	  --encoder-langtok "tgt" \
          --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
          --source-lang eng --target-lang rus \
          --beam 5   > "$OUTDIR"/test_engrus.log

python generate.py data-bin/ted_8_sepv/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --score-reference \
          --remove-bpe sentencepiece \
	  --encoder-langtok "tgt" \
          --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
          --source-lang eng --target-lang glg \
          --beam 5   > "$OUTDIR"/test_engglg.log

python generate.py data-bin/ted_8_sepv/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --score-reference \
          --remove-bpe sentencepiece \
	  --encoder-langtok "tgt" \
          --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
          --source-lang eng --target-lang por \
          --beam 5   > "$OUTDIR"/test_engpor.log

python generate.py data-bin/ted_8_sepv/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --score-reference \
          --remove-bpe sentencepiece \
	  --encoder-langtok "tgt" \
          --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
          --source-lang eng --target-lang slk \
          --beam 5   > "$OUTDIR"/test_engslk.log

python generate.py data-bin/ted_8_sepv/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --score-reference \
          --remove-bpe sentencepiece \
	  --encoder-langtok "tgt" \
          --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
          --source-lang eng --target-lang ces \
          --beam 5   > "$OUTDIR"/test_engces.log

#grep ^H checkpoints/tag_fw_slk-eng/fwtrans_test.log | cut -f3 > checkpoints/tag_fw_slk-eng/fwtrans_test.decode
