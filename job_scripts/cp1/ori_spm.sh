#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2330
#SBATCH --mem=15GB
#SBATCH --job-name=lorelei
#SBATCH --output=checkpoints/ori_noset0_spm8000_lstm/train-%j.out
#SBATCH --error=checkpoints/ori_noset0_spm8000_lstm/train-%j.error

python train.py data-bin/ori_noset0_spm8000_jd/ \
	--source-lang ori --target-lang eng \
	--task translation \
	--arch lstm_wiseman_iwslt_de_en \
	--max-tokens 1500 \
        --dropout 0.1 \
	--encoder-layers 2 \
	--decoder-layers 2 \
	--no-epoch-checkpoints \
	--keep-last-epochs 3 \
	--max-epoch 20 \
	--lr 0.001 \
	--optimizer adam \
	--share-all-embeddings \
	--distributed-world-size 1 \
	--save-dir checkpoints/ori_noset0_spm8000_lstm
