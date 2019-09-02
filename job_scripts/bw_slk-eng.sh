#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2330
#SBATCH --mem=15GB

#SBATCH --job-name=bw_slk-eng
#SBATCH --output=checkpoints/train_logs/bw_slk-eng_train-%j.out
#SBATCH --output=checkpoints/train_logs/bw_slk-eng_train-%j.err

python train.py data-bin/ted_slk_eng/ \
	--source-lang eng --target-lang slk \
	--task translation \
	--arch transformer_iwslt_de_en \
	--max-epoch 50 \
	--no-epoch-checkpoints \
	--distributed-world-size 1 \
	--share-all-embeddings \
	--dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 --weight-decay 0.0 \
	--left-pad-source False \
	--optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt' \
	--warmup-init-lr 1e-7 --warmup-updates 4000 --lr 1e-4 \
	--criterion 'label_smoothed_cross_entropy' --label-smoothing 0.1 \
	--max-tokens 1200 \
	--seed 2 \
	--log-interval 100 \
	--save-dir checkpoints/bw_slk-eng
