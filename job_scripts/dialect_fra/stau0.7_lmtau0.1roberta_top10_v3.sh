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

MODEL_DIR=checkpoints/dialect_fra/stau0.7_lmtau0.1roberta_top10_v3/
mkdir -p $MODEL_DIR

export PYTHONPATH="$(pwd)"

echo 'slurm id '$SLURM_JOB_ID >> $MODEL_DIR/train.log

#CUDA_VISIBLE_DEVICES=$1 python train.py data-bin/ted_eight_sepv/ \
#	  --dialect-tau 1  \
CUDA_VISIBLE_DEVICES=$1 python train.py data-bin/dialects/ \
	  --task translation \
	  --arch transformer_iwslt_de_en \
	  --max-epoch 60 \
          -s "fra" -t "eng" \
	  --source-tau 0.7  \
	  --dialect-tau 0.1  \
	  --lm-topk 10  \
    --eval-bleu --remove-bpe sentencepiece --sacrebleu \
	  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 --weight-decay 0.0 \
          --lm-path checkpoints/dialect_fra/mlm_from_robertabase_v1/checkpoint_best.pt \
          --lm-dict-path data-bin/dialects_framono/dict.txt \
	  --distributed-world-size 1 \
	  --share-all-embeddings \
	  --no-epoch-checkpoints \
	  --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt_decay' \
	  --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 2e-4  \
	  --criterion 'label_smoothed_cross_entropy' --label-smoothing 0.1 \
	  --max-tokens 2500 \
	  --update-freq 4 \
	  --seed 2 \
  	  --max-source-positions 150 --max-target-positions 150 \
  	  --save-dir $MODEL_DIR \
          --encoder-normalize-before --decoder-normalize-before \
          --scale-norm \
	  --log-interval 100 >> $MODEL_DIR/train.log 2>&1
	  #--log-interval 1
  #--utility-type 'ave' \
  #--data-actor 'ave_emb' \
  #--data-actor-multilin \
  #--update-language-sampling 2 \
  #--data-actor-model-embed  1 \
  #--data-actor-embed-grad 0 \
  #--out-score-type 'sigmoid' \
	#--log-interval 1 
  	#--sample-instance \
