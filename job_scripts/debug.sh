

python train.py data-bin/debug/ \
	--source-lang src --target-lang trg \
	--task translation \
	--arch lstm \
	--max-tokens 100 \
	--dataset-impl raw \
	--no-epoch-checkpoints \
	--save-dir checkpoints/debug 
