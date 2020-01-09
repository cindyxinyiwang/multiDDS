# Switchout: an Efficient Data Augmentation Algorithm for Multilingual Neural Machine Translation (Wang et al., 2018)

This page includes usage of the switchout method from the paper [Switchout: an Efficient Data Augmentation Algorithm for Multilingual Neural Machine Translation(Wang et al., 2018)](https://arxiv.org/abs/1808.07512).


## Example usage
```bash
fairseq-train \
    data-bin/wmt16_en_de_bpe32k \
    --source-tau 0.8 --target-tau 0.8 \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 
```

Note that the --source-tau and --target-tau parameters are the sampling temperature for the source and the target sentences. It should be a float number from [0, 1]. 
