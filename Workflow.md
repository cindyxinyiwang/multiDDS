### Interpretation of DDS repo
### Author: Weiting Tan, Date: Sept 15th, 2020


#### CLI Interface
- sample script from job-script folder shown below:
```
python train.py data-bin/ted_8_related/ \
	  --task multilingual_translation \
	  --arch multilingual_transformer_iwslt_de_en \
	  --max-epoch 40 \
          --dataset-type "multi" \
          --lang-pairs "aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng" \
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
          --datasize-t 1 \
	  --update-language-sampling 1000 \
  	  --data-actor 'base' \
  	  --data-actor-lr 0.0001 \
  	  --data-actor-optim-step 200 \
          --utility-type 'ave' \
          --datasize-t 1 \
          --pretrain-data-actor \
          --pretrain-type "datasize" \
	  --log-interval 100 >> $MODEL_DIR/train.log 2>&1
```
- pay special attention to the following:
    - arch `multilingual_transformer`, located in models folder, which is basically an extension on transformer model with litte variation to support multilingual trainining's setting
    - task `multilingual_translation`, extended on translation task, define the data loading, selection, etc.
    - `data-actor` controls the 


### Multilingual Transformer Model:
- Base model is FairseqMultiModel:
    - BaseFairseqModel:
        - build_model has to be extended and implemented
        - extract_feature: similar to forward but only return feature, `need a final map to vocab_size/output_dim`
        - max_position: controls dthe maxlen supported by the model
        - other functions including `make_generation_fast`, `from pretrained`, etc, not checked in detail and should not need to modify
    - FairseqMultiModel:
        - init receives an encoder and a decoder with args, which are both dictionary `{model_key: FairseqEncoder/Decoder}`
        - use EncoderDecoder model to connect each pair
        ```
        self.models = nn.ModuleDict({
            key: FairseqModel(encoders[key], decoders[key])
            for key in self.keys
        })
        ```
        - buiild_shared_embeddings:
            - takes in callable `build_embedding` function to return
            ``` 
            return build_embedding(
            shared_dict, embed_dim, pretrained_embed_path
            )
            ```
            - take dict of language and their dict/vocab_token mapping, use it as shared_dict for function above
            - the dict has to be a joint dictionary for share-embedding
            - forward is same as regular EncoderDecoder, just wraped in a loop for keys (language ids)
- build_model:
    - first take care of embedding, use a regular embedding layer and build it with shared_dict, embed_dim, padding_dimension as normal pytorch unit. Support an optional pretrained embedding as well
    - the shared embedding tokens are calculated using default MultiModel's build_shared_embedding as I described above, e.g:
    ```
    shared_encoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                dicts=task.dicts,
                langs=task.langs,
                embed_dim=args.encoder_embed_dim,
                build_embedding=build_embedding,
                pretrained_embed_path=args.encoder_embed_path,
            )
    ```
    - Encoder and Decoder are called from `TransformerEncoder` and `TransformerDecoder`, which is an default implemetation of paper `Attention is all you need`, no need to change the model architecture for this  research purpose but might try other archiecture that are LSTM based in the future.

    
### Train, Trainer, and Task:
- Train:
    - At first, the script will call the train.py script. train.py generally include following steps:
    - Pretrain Steps:
        - task.setup_task(args)
        - build model with  `model = task.build_model(args)`
        - build criterion with `criterion = task.build_criterion(args)`
    - Training steps:
        - build trainer: `trainer = Trainer(args, task, model, criterion)` 
        - load latest checkpoint if has any
        - optional: pretrain data_actor, unique to this paper's implementation
        - set up train meters
        - start training: (in a while loop, keep going until model converges or max-epoch/early stop reached)
            - call the train function: `epoch_itr = train(args, trainer, task, epoch_itr, generator, filtered_maxpos_indices)`
            - get validation loss: `valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets, generator)`
            - update learning rate: `lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])`
            - save checkpoint
    - train function: (controls training for 1 epoch only), largely re-written by the author of DDS paper
        - 