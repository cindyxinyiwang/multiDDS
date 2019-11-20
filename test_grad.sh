#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2330
#SBATCH --mem=15GB

#SBATCH --job-name=fw_slk-eng
#SBATCH --output=checkpoints/train_logs/fw_slk-eng_train-%j.out
#SBATCH --output=checkpoints/train_logs/fw_slk-eng_train-%j.err

export PYTHONPATH="$(pwd)"
#export CUDA_VISIBLE_DEVICES="2"

MODEL_DIR=checkpoints/test_grad/
mkdir -p $MODEL_DIR

export PYTHONPATH="$(pwd)"

CUDA_VISIBLE_DEVICES=$1 python train.py data-bin/ted_eight/ \
	--task multilingual_translation \
	--arch multilingual_transformer_iwslt_de_en \
	--max-epoch 50 \
  --dataset-type "multi" \
  --lang-pairs "aze-eng,tur-eng" \
  --eval-lang-pairs "aze-eng,tur-eng" \
  --save-interval-updates 2 \
	--no-epoch-checkpoints \
	--distributed-world-size 1 \
	--share-all-embeddings --share-decoders --share-encoders \
	--dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 --weight-decay 0.0 \
	--left-pad-source 'True' --left-pad-target 'False' \
	--optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt' \
	--warmup-init-lr 1e-7 --warmup-updates 4000 --lr 1e-4 \
	--criterion 'label_smoothed_cross_entropy'  \
	--max-tokens 1000 \
	--seed 2 \
	--log-interval 100 \
  --encoder-langtok 'tgt'  \
  --no-dev \
  --data-actor 'base' \
  --data-actor-lr 0.0001 \
  --scale-norm \
  --datasize-t 5 \
  --language-weight "1" \
  --utility-type 'median' \
  --data-actor 'base' \
  --pretrain-data-actor \
  --discount-grad \
  --update-language-sampling 2 \
          --pretrain-type "datasize" \
          --exact-update \
          --loss-weight "low" \
  --max-source-positions 1000 --max-target-positions 1000 \
  --remove-bpe sentencepiece \
  --save-dir $MODEL_DIR 
          #--data-actor-step-update \
  #--eval-bleu \
  #--update-language-sampling 2 \
  #--data-actor-model-embed  1 \
  #--data-actor-embed-grad 0 \
  #--out-score-type 'sigmoid' \
	#--log-interval 1 
  #--sample-instance \
