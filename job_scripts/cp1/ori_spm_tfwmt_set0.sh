#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --partition=learnfair
#SBATCH --time=2330
#SBATCH --mem=15GB
#SBATCH --output=checkpoints/ori_alignv1_spm8000_tfwmt/train-%j.out
#SBATCH --error=checkpoints/ori_alignv1_spm8000_tfwmt/train-%j.error

python train.py data-bin/ori_alignv1_spm8000_jd/ \
	--source-lang ori --target-lang eng \
	--task translation \
	--arch transformer_wmt_en_de \
	--max-tokens 800 \
	--max-source-positions 512 --max-target-positions 512 \
	--skip-invalid-size-inputs-valid-test \
	--lr 0.001 \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-7 \
        --warmup-updates 4000 \
        --lr 1e-4 \
        --clip-norm 0.0 \
        --dropout 0.1 \
        --attention-dropout 0.1 \
        --relu-dropout 0.1 \
        --weight-decay 0.0 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
	--max-epoch 20 \
	--share-all-embeddings \
	--save-dir checkpoints/ori_alignv1_spm8000_tfwmt
