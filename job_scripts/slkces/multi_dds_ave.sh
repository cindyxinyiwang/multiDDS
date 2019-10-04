#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2330
#SBATCH --mem=15GB

##SBATCH --job-name=fw_slk-eng
##SBATCH --output=checkpoints/train_logs/fw_slk-eng_train-%j.out
##SBATCH --output=checkpoints/train_logs/fw_slk-eng_train-%j.err

MODEL_DIR=checkpoints/slkces/multi_dds_ave/
mkdir -p $MODEL_DIR

#export PYTHONPATH="$(pwd)"

#CUDA_VISIBLE_DEVICES=$1 python train.py data-bin/ted_slkces/ \
python train.py data-bin/ted_slkces/ \
	--task multilingual_translation \
	--arch multilingual_transformer_iwslt_de_en \
  --dataset-type "multi" \
	--max-epoch 70 \
  --sample-instance \
  --lang-pairs "slk-eng,ces-eng,eng-slk,eng-ces" \
	--no-epoch-checkpoints \
	--distributed-world-size 1 \
	--share-all-embeddings --share-decoders --share-encoders \
	--dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 --weight-decay 0.0 \
	--left-pad-source 'True' --left-pad-target 'False' \
	--optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt' \
	--warmup-init-lr 1e-7 --warmup-updates 4000 --lr 1e-4 \
	--criterion 'label_smoothed_cross_entropy' --label-smoothing 0.1 \
	--max-tokens 2400 \
	--update-freq 2 \
	--max-tokens-valid 1200 \
	--seed 2 \
	--log-interval 100 \
  --encoder-langtok 'tgt'  \
  --max-source-positions 1000 --max-target-positions 1000 \
  --save-dir $MODEL_DIR \
  --data-actor-multilin \
  --utility-type 'ave' \
  --update-language-sampling 200 \
  --data-actor 'base' \
  --data-actor-lr 0.0001 \
  --data-actor-optim-step 200 \
	--log-interval 100 >> $MODEL_DIR/train.log 2>&1
