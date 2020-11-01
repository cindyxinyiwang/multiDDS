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
        - check dds_select, if use dds selection, filter training data 
        ```
        if epoch_itr.epoch == args.select_by_dds_epoch and args.select_by_dds_epoch > 0:
            epoch_itr, _ = trainer.get_filtered_train_iterator(epoch_itr.epoch, filtered_maxpos_indices=filtered_maxpos_indices)
        ```
        - in the process (train steps in one epoch):
            - call train_step from `trainer` which return the log_output
            - trainer call update_language_sampler => `use reinforcement to influence data sampling`
            ```
            log_output = trainer.train_step(samples, update_actor=update_actor)
            ```
            - get validate_loss and should_stop signal `valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets, generator)` and same the checkpoint (in original code, fairseq does not save checkpoint here)

- Trainer:
    - in train.py, two function from traner is used: `get_filtered_train_iterator` and `train_step`, I will give detailed explanation of these two functions and their related utilities. The other important functions include `pretrain_data_actor`, ` update_language_sampler`, `update_language_sampler_multilin`, etc.
    - init:
        - define args, task, meters, etc. 
        - check which data actor to use: (the author implemented four customized data actor):
            - BaseActor: only contains a linear layer which return logits of dimension [language_size] => a distribution of data to load for each language
            - LanguageActor: using an embedding to convert language into an embedding_dimension defined by user and then project back to a distribution of languages
            - AveEmbActor: average embedding actor, use the source and target dictionary from task. Embed the source and target word, concat them and project to a dimension [#languages], representing the data distribution. This method takes the actual tokens into account.
        - set dev set iterator if data_actor_step_update is set, 
        ```
        self.dev_itr = self.task.get_batch_iterator(
            dataset=self.task.dataset('valid'),....
            )[0]
        ```
    - get_train_iterator:
        - return EpochBatchIterator, bascially the same as the default implemetation by fairseq, use `task.get_batch_iterator` which uses the `fairseq_task`'s function (the author modified source code)
        - details of modified get_batch_iterator:
            - input expanded with `data_actor=None, trainer=None, data_filter_percentage=-1, filtered_maxpos_indices=None,`
            - implement data filtering with data_utils:
            ```
            if data_filter_percentage > 0:
                indices = data_utils.filter_by_data_actor(indices, dataset, data_actor, data_filter_percentage, trainer=trainer)
            ```
            - filter_by_data_actor(indices, dataset, data_actor, data_filter_percentage=-1, trainer=None):
                - if not random data filter, call data actor to get a score (a distribution of data input of each language)
                ```
                itr = iterators.EpochBatchIterator(
                    dataset=dataset,
                    collate_fn=dataset.collater,
                    batch_sampler=batch_sampler
                ).next_epoch_itr(shuffle=False)

                idx_start, idx_end = 0, 0
                scores = np.zeros(len(indices))
                ids = np.zeros(len(indices), dtype=np.int64)

                for i, sample in enumerate(itr):
                    sample = trainer._prepare_sample(sample)
                    sample = list(sample.values())[0]

                    # score is of size B X 1
                    score = data_actor(sample['net_input']['src_tokens'], sample['target']).data.cpu().numpy()
                    idx_start = idx_end

                    # update the batch range
                    idx_end = idx_start + score.shape[0]
                    scores[idx_start:idx_end] = score.ravel()
                    ids[idx_start:idx_end] = sample['id'].data.cpu().numpy().ravel()

                # argsort is ascending order
                preserved_indices = np.argsort(scores)[int(len(indices)*data_filter_percentage):]
                indices = np.array(ids)[preserved_indices]
        ```
    - get_filtered_train_iterator:
        - basically the same as function above, with one extra parameter:`data_filter_percentage=self.args.data_filter_percentage`
        - adding the percentage will make the `get_train_iterator` call the data_actor_filter function shown above
    - train_step:
        - `train_step(self, samples, dummy_batch=False, raise_oom=False, update_actor=True)`
        - in side train_step, the task's train_step is called, passed in the sample, model, criterion, optimizer , etc.
        ```
        loss, sample_size, logging_output = self.task.train_step(
            sample, self.model, self.criterion, self.optimizer,
            ignore_grad, data_actor=data_actor, 
            loss_copy=cached_loss, data_actor_out=data_actor_out,
        )
        ```
        - then save the training gradient
        - if try to update data_actor:
            - use model to train on the valid sample and get valid's loss, etc
            - train again on the sample and get `current loss and cached loss` to update the reward `reward = 1./eta * (cur_loss[k] - cached_loss[k])` and backward on the data_actor's network

    - update_language_sampler:
        - load in optimizer, data_actor, data_optimizer
        - like normal training, call task.train_step and save gradients. Then train on valid dataset and get the gradient, update the data actor as well as a simlarity probability => change the distribution of training set based the distribution
        ```
        self.task.dataset('train').update_sampling_distribution(sim_list)
        ```
        - seems like this function has error in variable name mismatch??? and where is the update_sampling_distribution?
        
### Modification on data selection by data actor
- Following num reset needed or not? The update_frequency and num reset work together. When resetting
the epoch iterator, epoch_itr.next_epoch_itr is called with an `
offset=reset_idx*(args.update_language_sampling*args.update_freq[0]+1)`
```
   if args.update_language_sampling > 0 and args.select_by_dds_epoch < 0 and (not args.data_actor_step_update):
        num_reset = len(epoch_itr.frozen_batches) // (args.update_language_sampling*args.update_freq[0]+1)
        datasize = args.update_language_sampling*args.update_freq[0]+1
        if num_reset * datasize < len(epoch_itr.frozen_batches):
            num_reset += 1
    else:
        num_reset = 1
        datasize = -1
```
- 