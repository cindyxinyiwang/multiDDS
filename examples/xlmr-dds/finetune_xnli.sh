TOTAL_NUM_UPDATES=2036  
WARMUP_UPDATES=122      
LR=2e-05                
NUM_CLASSES=3
MAX_SENTENCES=16        
ROBERTA_PATH=pretrained_models/xlrm.large/model.pt 
MODEL_DIR=checkpoints/xnli/uniform/

CUDA_VISIBLE_DEVICES=$1 python train.py data-bin/xnli/ \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 4400 \
    --task multilingual_sentence_prediction \
    --langs "ar,bg,de" \
    --dataset-type "multi" \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --max-epoch 10 \
    --find-unused-parameters \
    --save-dir $MODEL_DIR \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --log-interval 100 
    #--fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
