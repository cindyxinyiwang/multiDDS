#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --mem=15GB

OUTDIR=$1

python generate.py data-bin/ted_8_related/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	  --sacrebleu \
          --lang-pairs "aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng" \
          --source-lang aze --target-lang eng \
          --beam 5   > "$OUTDIR"/test_azeeng.log

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
          --beam 5   > "$OUTDIR"/test_tureng.log

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
          --beam 5   > "$OUTDIR"/test_beleng.log

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
          --beam 5   > "$OUTDIR"/test_ruseng.log

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
          --beam 5   > "$OUTDIR"/test_glgeng.log

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
          --beam 5   > "$OUTDIR"/test_poreng.log

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
          --beam 5   > "$OUTDIR"/test_slkeng.log

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
          --beam 5   > "$OUTDIR"/test_ceseng.log

