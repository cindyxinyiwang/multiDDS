

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="1"
OUTDIR=checkpoints/m2o_ted_eight/tcs_aze/

python generate.py data-bin/ted_eight/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 16 \
          --lenpen 1.5 \
          --remove-bpe sentencepiece \
          --lang-pairs "aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng" \
          --encoder-langtok 'tgt'  \
          --source-lang aze --target-lang eng \
          --skip-invalid-size-inputs-valid-test \
          --beam 5   > "$OUTDIR"/test_azeeng.log

python generate.py data-bin/ted_eight/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 16 \
          --lenpen 1.5 \
          --remove-bpe sentencepiece \
          --lang-pairs "aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng" \
          --encoder-langtok 'tgt'  \
          --source-lang tur --target-lang eng \
          --skip-invalid-size-inputs-valid-test \
          --beam 5   > "$OUTDIR"/test_tureng.log

python generate.py data-bin/ted_eight/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 16 \
          --lenpen 1.5 \
          --remove-bpe sentencepiece \
          --lang-pairs "aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng" \
          --encoder-langtok 'tgt'  \
          --source-lang bel --target-lang eng \
          --skip-invalid-size-inputs-valid-test \
          --beam 5   > "$OUTDIR"/test_beleng.log

python generate.py data-bin/ted_eight/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 16 \
          --lenpen 1.5 \
          --remove-bpe sentencepiece \
          --lang-pairs "aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng" \
          --encoder-langtok 'tgt'  \
          --source-lang rus --target-lang eng \
          --skip-invalid-size-inputs-valid-test \
          --beam 5   > "$OUTDIR"/test_ruseng.log

python generate.py data-bin/ted_eight/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 16 \
          --lenpen 1.5 \
          --remove-bpe sentencepiece \
          --lang-pairs "aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng" \
          --encoder-langtok 'tgt'  \
          --source-lang glg --target-lang eng \
          --skip-invalid-size-inputs-valid-test \
          --beam 5   > "$OUTDIR"/test_glgeng.log

python generate.py data-bin/ted_eight/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 16 \
          --lenpen 1.5 \
          --remove-bpe sentencepiece \
          --lang-pairs "aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng" \
          --encoder-langtok 'tgt'  \
          --source-lang por --target-lang eng \
          --skip-invalid-size-inputs-valid-test \
          --beam 5   > "$OUTDIR"/test_poreng.log

python generate.py data-bin/ted_eight/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 16 \
          --lenpen 1.5 \
          --remove-bpe sentencepiece \
          --lang-pairs "aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng" \
          --encoder-langtok 'tgt'  \
          --source-lang slk --target-lang eng \
          --skip-invalid-size-inputs-valid-test \
          --beam 5   > "$OUTDIR"/test_slkeng.log

python generate.py data-bin/ted_eight/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 16 \
          --lenpen 1.5 \
          --remove-bpe sentencepiece \
          --lang-pairs "aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng" \
          --encoder-langtok 'tgt'  \
          --source-lang ces --target-lang eng \
          --skip-invalid-size-inputs-valid-test \
          --beam 5   > "$OUTDIR"/test_ceseng.log


#grep ^H checkpoints/tag_fw_slk-eng/fwtrans_test.log | cut -f3 > checkpoints/tag_fw_slk-eng/fwtrans_test.decode
