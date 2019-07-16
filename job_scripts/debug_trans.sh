#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2330
#SBATCH --mem=15GB

#SBATCH --job-name=lorelei
#SBATCH --output=checkpoints/debug_sde/decode.out
#SBATCH --output=checkpoints/debug_sde/decode-%j.err

python generate.py data-bin/debug \
	--path checkpoints/debug_sde/checkpoint_best.pt \
	--source-lang src --target-lang trg \
	--skip-invalid-size-inputs-valid-test \
	--gen-subset test \
	--dataset-impl raw \
	--sde \
	--task translation \
	--nbest 1 \
	--beam 10 \
	--lenpen 1 \
	--prefix-size 0 \
	--batch-size 1 \
	--max-source-positions 10000 --max-target-positions 10000 \
	--max-len-b 150 \
	--remove-bpe "@@ " \
	--max-len-a 0 

