#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --mem=15GB

##SBATCH --job-name=fw_slk-eng
##SBATCH --output=checkpoints/train_logs/fw_slk-eng_train-%j.out
##SBATCH --output=checkpoints/train_logs/fw_slk-eng_train-%j.err

#export PYTHONPATH="$(pwd)"
#export CUDA_VISIBLE_DEVICES="2"

MODEL_DIR=checkpoints/o2m_ted_eight/uniform_dds_featvalloss/
mkdir -p $MODEL_DIR

export PYTHONPATH="$(pwd)"

#python train.py data-bin/ted_eight/ \
CUDA_VISIBLE_DEVICES=$1 python train.py data-bin/ted_eight/ \
	  --task multilingual_translation \
	  --arch multilingual_transformer_iwslt_de_en \
	  --max-epoch 50 \
    --dataset-type "multi" \
    --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
	  --no-epoch-checkpoints \
	  --distributed-world-size 1 \
  	--encoder-langtok 'tgt'  \
	  --share-all-embeddings --share-decoders --share-encoders \
	  --dropout 0.3 --attention-dropout 0.3 --relu-dropout 0.3 --weight-decay 0.0 \
	  --left-pad-source 'True' --left-pad-target 'False' \
	  --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt_decay' \
	  --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 2e-4 --lr-shrink 0.8 \
	  --criterion 'label_smoothed_cross_entropy' --label-smoothing 0.1 \
	  --max-tokens 2400 \
	  --update-freq 4 \
	  --seed 2 \
  	--max-source-positions 150 --max-target-positions 150 \
  	--save-dir $MODEL_DIR \
    --encoder-normalize-before --decoder-normalize-before \
    --scale-norm \
  	--update-language-sampling 500 \
  	--data-actor 'base' \
  	--data-actor-lr 0.0001 \
  	--data-actor-optim-step 1 \
	  --no-dev \
	  --data-actor-multilin \
    --utility-type 'ave' \
    --datasize-t 1 \
    --pretrain-type "datasize" \
    --pretrain-data-actor \
    --feature-type "valid_loss" \
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
