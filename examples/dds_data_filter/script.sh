#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --mem=15GB

#export PYTHONPATH="$(pwd)"
#export CUDA_VISIBLE_DEVICES="2"

MODEL_DIR=checkpoints/m2o_ted_eight_sepv/uniform_dds_eight/
mkdir -p $MODEL_DIR

export PYTHONPATH="$(pwd)"

python train.py data-bin/debug/ \
	  --task multilingual_translation \
	  --arch multilingual_transformer_iwslt_de_en \
	  --max-epoch 50 \
          --dataset-type "multi" \
          --lang-pairs "src-trg" \
	  --no-epoch-checkpoints \
	  --distributed-world-size 1 \
	  --share-decoder-input-output-embed --share-decoders --share-encoders \
	  --dropout 0.3 --attention-dropout 0.3 --relu-dropout 0.3 --weight-decay 0.0 \
	  --left-pad-source 'True' --left-pad-target 'False' \
	  --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt_decay' \
	  --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 2e-4 --lr-shrink 0.8 \
	  --criterion 'label_smoothed_cross_entropy' --label-smoothing 0.1 \
	  --max-tokens 4800 \
	  --update-freq 2 \
	  --seed 2 \
  	  --max-source-positions 150 --max-target-positions 150 \
  	  --save-dir $MODEL_DIR \
          --encoder-normalize-before --decoder-normalize-before \
          --scale-norm \
	  --update-language-sampling 200 \
  	  --data-actor 'base' \
  	  --data-actor-lr 0.0001 \
  	  --data-actor-optim-step 200 \
	  --no-dev \
	  --data-actor-multilin \
          --utility-type 'ave' \
	  --log-interval 100 >> $MODEL_DIR/train.log 2>&1
  #--utility-type 'ave' \
  #--data-actor 'ave_emb' \
  #--data-actor-multilin \
  #--update-language-sampling 2 \
  #--data-actor-model-embed  1 \
  #--data-actor-embed-grad 0 \
  #--out-score-type 'sigmoid' \
	#--log-interval 1 
  #--sample-instance \
