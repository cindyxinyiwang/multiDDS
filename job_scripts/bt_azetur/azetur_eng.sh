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

MODEL_DIR=checkpoints/bt_azetur/azetur_eng/
mkdir -p $MODEL_DIR

export PYTHONPATH="$(pwd)"

echo 'slurm id '$SLURM_JOB_ID >> $MODEL_DIR/train.log

#CUDA_VISIBLE_DEVICES=$1 python train.py data-bin/ted_eight_sepv/ \
python train.py data-bin/bt_azetur/ \
	  --task multilingual_translation \
	  --arch multilingual_transformer_iwslt_de_en \
	  --max-epoch 50 \
          --dataset-type "round_robin" --upsample-factor 1 \
          --lang-pairs "aze-eng,tur-eng" \
          --eval-lang-pairs "aze-eng" \
	  --distributed-world-size 1 \
	  --share-all-embeddings --share-decoders --share-encoders \
	  --no-epoch-checkpoints \
	  --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt_decay' \
	  --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 2e-4  \
	  --criterion 'label_smoothed_cross_entropy' --label-smoothing 0.1 \
	  --max-tokens 4500 \
	  --update-freq 2 \
	  --seed 2 \
  	  --max-source-positions 150 --max-target-positions 150 \
  	  --save-dir $MODEL_DIR \
          --encoder-normalize-before --decoder-normalize-before \
          --scale-norm \
	  --log-interval 100 >> $MODEL_DIR/train.log 2>&1
	  #--sample-instance \
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
