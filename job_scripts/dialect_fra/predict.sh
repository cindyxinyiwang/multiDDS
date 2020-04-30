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

MODEL_DIR=checkpoints/dialect_fra/prediction/
mkdir -p $MODEL_DIR

export PYTHONPATH="$(pwd)"

echo 'slurm id '$SLURM_JOB_ID >> $MODEL_DIR/train.log

#	  --classification-head-name "fra" \
#CUDA_VISIBLE_DEVICES=$1 python train.py data-bin/ted_eight_sepv/ \
CUDA_VISIBLE_DEVICES=$1 python train.py data-bin/dialect_fra_predict/ \
	  --task sentence_prediction \
	  --arch roberta_small \
	  --criterion sentence_prediction \
	  --num-classes 2 \
	  --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
	  --max-epoch 50 \
	  --distributed-world-size 1 \
	  --no-epoch-checkpoints \
	  --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt_decay' \
	  --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 2e-4  \
	  --max-tokens 4500 --max-positions 512  \
	  --update-freq 2 \
	  --seed 2 \
  	  --save-dir $MODEL_DIR \
	  --log-interval 100 >> $MODEL_DIR/train.log 2>&1
	  #--log-interval 1
          #--no-epoch-checkpoints \
  #--utility-type 'ave' \
  #--data-actor 'ave_emb' \
  #--data-actor-multilin \
  #--update-language-sampling 2 \
  #--data-actor-model-embed  1 \
  #--data-actor-embed-grad 0 \
  #--out-score-type 'sigmoid' \
	#--log-interval 1 
  	#--sample-instance \
