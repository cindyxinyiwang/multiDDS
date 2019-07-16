

python train.py data-bin/debug/ \
	--source-lang src --target-lang trg \
	--sde \
	--distributed-world-size 1 \
	--task translation \
	--arch transformer \
	--max-tokens 100 \
	--dataset-impl raw \
	--no-epoch-checkpoints \
	--lr 0.001 \
	--distributed-world-size 1 \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-7 \
        --warmup-updates 4000 \
        --lr 1e-4 \
        --clip-norm 0.0 \
        --dropout 0.1 \
        --attention-dropout 0.1 \
        --relu-dropout 0.1 \
        --weight-decay 0.0 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
	--save-dir checkpoints/debug_tf_sde 
