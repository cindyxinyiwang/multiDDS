#!/bin/bash

MODEL_DIR=checkpoints/debug/switchout/
mkdir -p $MODEL_DIR

python train.py data-bin/debug/  --source-lang src --target-lang trg \
	  --task translation \
	  --arch transformer \
          --source-tau 0.8 \
          --target-tau 0.8 \
	  --max-epoch 50 \
	  --no-epoch-checkpoints \
	  --distributed-world-size 1 \
	  --share-all-embeddings \
	  --dropout 0.3 --attention-dropout 0.3 --relu-dropout 0.3 --weight-decay 0.0 \
	  --left-pad-source 'True' --left-pad-target 'False' \
	  --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt_decay' \
	  --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 2e-4 --lr-shrink 0.8 \
	  --criterion 'label_smoothed_cross_entropy' --label-smoothing 0.1 \
	  --max-tokens 9600 \
	  --seed 2 \
  	  --max-source-positions 150 --max-target-positions 150 \
  	  --save-dir $MODEL_DIR \
	  --log-interval 100 
