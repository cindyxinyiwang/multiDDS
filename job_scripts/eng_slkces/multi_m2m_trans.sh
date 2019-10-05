

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="1"
OUTDIR=checkpoints/eng_slkces/multi_m2m/

python generate.py data-bin/ted_slkces/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 16 \
          --lenpen 1.5 \
          --remove-bpe sentencepiece \
          --lang-pairs "eng-slk,eng-ces" \
          --encoder-langtok 'tgt'  \
          --source-lang eng --target-lang slk \
          --skip-invalid-size-inputs-valid-test \
          --beam 5   > "$OUTDIR"/test_engslk.log


python generate.py data-bin/ted_slkces/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 16 \
          --lenpen 1.5 \
          --remove-bpe sentencepiece \
          --lang-pairs "eng-slk,eng-ces" \
          --encoder-langtok 'tgt'  \
          --source-lang eng --target-lang ces \
          --skip-invalid-size-inputs-valid-test \
          --beam 5   > "$OUTDIR"/test_engces.log


#grep ^H checkpoints/tag_fw_slk-eng/fwtrans_test.log | cut -f3 > checkpoints/tag_fw_slk-eng/fwtrans_test.decode
