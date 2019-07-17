#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=dev
#SBATCH --time=2330
#SBATCH --mem=15GB
#SBATCH --output=checkpoints/ori_alignv1_sde/train-%j.out
#SBATCH --error=checkpoints/ori_alignv1_sde/train-%j.error

python train.py data-bin/ori_alignv1_sde/ \
	--source-lang ori --target-lang eng \
	--task translation \
	--arch lstm_wiseman_iwslt_de_en \
	--sde \
	--dataset-impl raw \
	--max-tokens 1100 \
	--max-source-positions 500 \
	--max-target-positions 500 \
        --dropout 0.1 \
	--encoder-layers 2 \
	--decoder-layers 2 \
	--no-epoch-checkpoints \
	--keep-last-epochs 3 \
	--max-epoch 20 \
	--lr 0.001 \
	--optimizer adam \
	--save-dir checkpoints/ori_alignv1_sde
