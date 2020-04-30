#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --mem=15GB
##SBATCH --exclude=compute-0-26

##SBATCH --job-name=fw_slk-eng
##SBATCH --output=checkpoints/train_logs/fw_slk-eng_train-%j.out
##SBATCH --output=checkpoints/train_logs/fw_slk-eng_train-%j.err

#export PYTHONPATH="$(pwd)"
#export CUDA_VISIBLE_DEVICES="2"

ROBERTA_PATH=checkpoints/roberta.base/model.pt
MODEL_DIR=checkpoints/dialect_fra/mlm_from_robertabase_v1/
mkdir -p $MODEL_DIR

export PYTHONPATH="$(pwd)"

echo 'slurm id '$SLURM_JOB_ID >> $MODEL_DIR/train.log

#python train.py data-bin/ted_eight_sepv/ \
CUDA_VISIBLE_DEVICES=$1 python train.py data-bin/dialects_framono/ \
    --restore-file $ROBERTA_PATH \
	  --task masked_lm \
	  --criterion masked_lm \
	  --arch roberta_base \
    --sample-break-mode complete --tokens-per-sample 512 \
	  --max-epoch 200 \
	  --distributed-world-size 1 \
	  --no-epoch-checkpoints \
	  --optimizer 'adam' --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 \
    --lr-scheduler 'polynomial_decay' --lr 1e-4 --warmup-updates 2000  \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
	  --max-sentences 4  --update-freq 8 \
	  --seed 2 \
  	--save-dir $MODEL_DIR \
	  --log-interval 100 >> $MODEL_DIR/train.log 2>&1
	  #--log-interval 1
