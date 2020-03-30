#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --mem=15GB
#SBATCH --exclude=compute-0-26

##SBATCH --job-name=fw_slk-eng
##SBATCH --output=checkpoints/train_logs/fw_slk-eng_train-%j.out
##SBATCH --output=checkpoints/train_logs/fw_slk-eng_train-%j.err

#exrust PYTHONPATH="$(pwd)"
#exrust CUDA_VISIBLE_DEVICES="2"

MODEL_DIR=checkpoints/bt_belrus/bel_up1_dds_m_dbase_lr1e-5_dis0.99_v1/
mkdir -p $MODEL_DIR

export PYTHONPATH="$(pwd)"

echo 'slurm id '$SLURM_JOB_ID >> $MODEL_DIR/train.log

#CUDA_VISIBLE_DEVICES=$1 python train.py data-bin/ted_eight_sepv/ \
CUDA_VISIBLE_DEVICES=$1 python train.py data-bin/bt_belrus/ \
	  --task bt_translation \
	  --arch multilingual_transformer_iwslt_de_en \
	  --max-epoch 50 \
          --dataset-type "round_robin" --upsample-factor 1 \
          --lang-pairs "bel-eng" \
          --eval-lang-pairs "bel-eng" \
	  --reset-optimizer --reset-meters --reset-dataloader \
    --only-optim-model-key "bel-eng" \
    --share-all-langpair-embeddings \
	  --eval-bleu --remove-bpe sentencepiece --sacrebleu \
	  --dropout 0.3 --attention-dropout 0.3 --relu-dropout 0.3 --weight-decay 0.0 \
	  --no-epoch-checkpoints \
    --bt-norm-by-word \
    --bt-optimizer-momentum 0.9 --bt-optimizer-nesterov \
    --bt_dds --discount-baseline-size 200 --discount-reward 0.99 \
    --lambda-denoising-config '1' \
    --reward-scale 1 \
    --data-actor-lr 1e-5 \
    --sampling --sampling-topk 10 --temperature 1 \
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
