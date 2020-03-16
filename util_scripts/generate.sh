


python interactive.py data-bin/debug/ \
       	--path checkpoints/debug/checkpoint_last.pt \
	--batch-size 1 \
	--buffer-size 2 \
	--beam 5  < data-bin/debug/test.src-trg.trg > ./trans_results/debug.log

grep ^H ./trans_results/debug.log | cut -f3 > ./trans_results/debug.decode


