

python train.py data-bin/debug/ \
	--source-lang src --target-lang trg \
	--task translation \
	--arch lstm \
	--max-tokens 100 \
	--dataset-impl raw \
	--no-epoch-checkpoints \
	--lr 0.001 \
	--distributed-world-size 1 \
	--sde \
	--save-dir checkpoints/debug_sde 
