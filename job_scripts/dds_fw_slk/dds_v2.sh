#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --mem=15GB
#SBATCH --exclude=compute-0-26

##SBATCH --job-name=fw_slk-eng
##SBATCH --output=checkpoints/train_logs/fw_slk-eng_train-%j.out
##SBATCH --output=checkpoints/train_logs/fw_slk-eng_train-%j.err

#export PYTHONPATH="$(pwd)"
#export CUDA_VISIBLE_DEVICES="2"

MODEL_DIR=checkpoints/dds_fw_slk/dds_v2/
mkdir -p $MODEL_DIR

export PYTHONPATH="$(pwd)"

echo 'slurm id '$SLURM_JOB_ID >> $MODEL_DIR/train.log

#CUDA_VISIBLE_DEVICES=$1 python train.py data-bin/ted_eight_sepv/ \
#	  --lambda-dds-config "0:0,2500:0,5000:1000" \
CUDA_VISIBLE_DEVICES=$1 python train.py data-bin/bt_slkces/ \
	  --task dds_translation \
	  --arch transformer_iwslt_de_en \
	  --max-epoch 50 \
          --share-all-embeddings \
	  -s slk -t eng \
	  --no-epoch-checkpoints \
	  --dropout 0.3 --weight-decay 0.0 \
	  --eval-bleu --remove-bpe sentencepiece --sacrebleu \
          --sampling --sampling-topk 10 --temperature 1 \
	  --lambda-dds-config "0:0,2000:0,3000:1" \
	  --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt_decay' \
	  --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 2e-4  \
	  --criterion 'label_smoothed_cross_entropy' --label-smoothing 0.1 \
	  --max-tokens 2000 \
	  --update-freq 4 \
	  --seed 2 \
  	  --max-source-positions 150 --max-target-positions 150 \
  	  --save-dir $MODEL_DIR \
          --encoder-normalize-before --decoder-normalize-before \
          --scale-norm \
	  --log-interval 100 >> $MODEL_DIR/train.log 2>&1
	  #--log-interval 1

