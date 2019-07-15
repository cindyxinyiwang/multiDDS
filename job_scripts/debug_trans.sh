#!/bin/bash

#SBATCH --partition=uninterrupted
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=2330
#SBATCH --mem=100GB


#SBATCH --job-name=decode_eli_transformer
#SBATCH --output=eli-qd-trans-%j.out
#SBATCH --error=eli-qd-trans-%j.er
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

