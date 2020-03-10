#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --mem=15GB
#SBATCH --exclude=compute-0-26

##SBATCH --job-name=fw_slk-eng
##SBATCH --output=checkpoints/train_logs/fw_slk-eng_train-%j.out
##SBATCH --output=checkpoints/train_logs/fw_slk-eng_train-%j.err

#export PYTHONPATH="$(pwd)"
#export CUDA_VISIBLE_DEVICES="$2"

OUTDIR=$1

python generate.py data-bin/bt_azetur/ \
          --task bt_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
          --lang-pairs "aze-eng" \
          --source-lang aze --target-lang eng \
          --beam 5   > "$OUTDIR"/test_b5.log
          #--task bt_translation \


python generate.py data-bin/bt_azetur/ \
          --task bt_translation \
          --gen-subset valid \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
          --lang-pairs "aze-eng" \
          --source-lang aze --target-lang eng \
          --beam 5   > "$OUTDIR"/valid_b5.log
          #--task bt_translation \

#grep ^H checkpoints/tag_fw_slk-eng/fwtrans_test.log | cut -f3 > checkpoints/tag_fw_slk-eng/fwtrans_test.decode
