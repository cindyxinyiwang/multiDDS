

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="1"
OUTDIR=checkpoints/slkces/multi_m2m_dds/

python generate.py data-bin/ted_slkces/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 16 \
          --lenpen 1.5 \
          --remove-bpe sentencepiece \
          --lang-pairs "slk-eng,ces-eng,eng-slk,eng-ces" \
          --encoder-langtok 'tgt'  \
          --source-lang slk --target-lang eng \
          --skip-invalid-size-inputs-valid-test \
          --beam 5   > "$OUTDIR"/test_slkeng.log

python generate.py data-bin/ted_slkces/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 16 \
          --lenpen 1.5 \
          --remove-bpe sentencepiece \
          --lang-pairs "slk-eng,ces-eng,eng-slk,eng-ces" \
          --encoder-langtok 'tgt'  \
          --source-lang ces --target-lang eng \
          --skip-invalid-size-inputs-valid-test \
          --beam 5   > "$OUTDIR"/test_ceseng.log


python generate.py data-bin/ted_slkces/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 16 \
          --lenpen 1.5 \
          --remove-bpe sentencepiece \
          --lang-pairs "slk-eng,ces-eng,eng-slk,eng-ces" \
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
          --lang-pairs "slk-eng,ces-eng,eng-slk,eng-ces" \
          --encoder-langtok 'tgt'  \
          --source-lang eng --target-lang ces \
          --skip-invalid-size-inputs-valid-test \
          --beam 5   > "$OUTDIR"/test_engces.log


#grep ^H checkpoints/tag_fw_slk-eng/fwtrans_test.log | cut -f3 > checkpoints/tag_fw_slk-eng/fwtrans_test.decode
