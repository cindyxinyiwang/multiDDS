#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2330
#SBATCH --mem=15GB
#SBATCH --job-name=lorelei
#SBATCH --output=checkpoints/ori_noset0_spm8000_sde/train-%j.out
#SBATCH --error=checkpoints/ori_noset0_spm8000_sde/train-%j.error

python train.py data-bin/ori_noset0_sde/ \
	--source-lang ori --target-lang eng \
	--task translation \
	--arch lstm \
	--sde \
	--dataset-impl raw \
	--max-tokens 1500 \
        --dropout 0.1 \
	--no-epoch-checkpoints \
	--keep-last-epochs 3 \
	--max-epoch 20 \
	--lr 0.001 \
	--distributed-world-size 1 \
	--save-dir checkpoints/ori_noset0_spm8000_sde
