#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2330
#SBATCH --mem=15GB

#SBATCH --job-name=lorelei
#SBATCH --output=checkpoints/debug_sde/eli-qd-trans-%j.out
#SBATCH --output=checkpoints/debug_sde/eli-qd-trans-%j.err

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
