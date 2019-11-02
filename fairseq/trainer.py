# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a network across multiple GPUs.
"""

from collections import OrderedDict
import contextlib
from itertools import chain
import math
import os
import sys
import numpy as np

import torch

from fairseq import checkpoint_utils, distributed_utils, models, optim, utils
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.optim import lr_scheduler
from fairseq.models import BaseActor, AveEmbActor

class Trainer(object):
    """Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, args, task, model, criterion, dummy_batch=None, oom_batch=None):
        self.args = args
        self.task = task

        # copy model and criterion to current device
        self._criterion = criterion
        self._model = model
        self.cuda = torch.cuda.is_available() and not args.cpu
        if args.fp16:
            self._criterion = self._criterion.half()
            self._model = self._model.half()
        if self.cuda:
            self._criterion = self._criterion.cuda()
            self._model = self._model.cuda()

        self._dummy_batch = dummy_batch
        self._oom_batch = oom_batch or dummy_batch

        self._lr_scheduler = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._prev_grad_norm = None
        self._wrapped_criterion = None
        self._wrapped_model = None

        self.init_meters(args)
        self.pretrained = False
        if self.args.data_actor == 'base' or self.args.data_actor == 'base_weight':
            # use language level data selector with only bias
            self.data_actor = BaseActor(args, len(self.args.lang_pairs))
            if self.cuda:
                self.data_actor = self.data_actor.cuda()
            self.data_optimizer = torch.optim.Adam([p for p in self.data_actor.parameters() if p.requires_grad], lr=self.args.data_actor_lr)
        elif self.args.data_actor == 'ave_emb':
            if self.args.data_actor_model_embed:
                self.data_actor = AveEmbActor(args, task, emb=self._model.embed_tokens)
            else:
                self.data_actor = AveEmbActor(args, task)
            if self.cuda:
                self.data_actor = self.data_actor.cuda()
            if not args.data_actor_embed_grad:
                self.data_optimizer = torch.optim.Adam([p for p in self.data_actor.project_out.parameters() if p.requires_grad], lr=self.args.data_actor_lr)
            else:
                self.data_optimizer = torch.optim.Adam([p for p in self.data_actor.parameters() if p.requires_grad], lr=self.args.data_actor_lr)
        else:
            self.data_actor = None
            self.data_optimizer = None
        
        if self.args.extra_data_actor == 'ave_emb':
            if self.args.data_actor_model_embed:
                self.extra_data_actor = AveEmbActor(args, task, emb=self._model.embed_tokens)
            else:
                self.extra_data_actor = AveEmbActor(args, task)
            if self.cuda:
                self.extra_data_actor = self.extra_data_actor.cuda()
            if not args.data_actor_embed_grad:
                self.extra_data_optimizer = torch.optim.Adam([p for p in self.extra_data_actor.project_out.parameters() if p.requires_grad], lr=self.args.data_actor_lr)
            else:
                self.extra_data_optimizer = torch.optim.Adam([p for p in self.extra_data_actor.parameters() if p.requires_grad], lr=self.args.data_actor_lr)
        else:
            self.extra_data_actor = None
            self.extra_data_optimizer = None
        self.baseline = None

    def init_meters(self, args):
        self.meters = OrderedDict()
        self.meters['train_loss'] = AverageMeter()
        self.meters['train_nll_loss'] = AverageMeter()
        self.meters['valid_loss'] = AverageMeter()
        self.meters['valid_nll_loss'] = AverageMeter()
        self.meters['wps'] = TimeMeter()       # words per second
        self.meters['ups'] = TimeMeter()       # updates per second
        self.meters['wpb'] = AverageMeter()    # words per batch
        self.meters['bsz'] = AverageMeter()    # sentences per batch
        self.meters['gnorm'] = AverageMeter()  # gradient norm
        self.meters['clip'] = AverageMeter()   # % of updates clipped
        self.meters['oom'] = AverageMeter()    # out of memory
        if args.fp16:
            self.meters['loss_scale'] = AverageMeter()  # dynamic loss scale
        self.meters['wall'] = TimeMeter()      # wall time in seconds
        self.meters['train_wall'] = StopwatchMeter()  # train wall time in seconds

    @property
    def criterion(self):
        if self._wrapped_criterion is None:
            if (
                utils.has_parameters(self._criterion)
                and self.args.distributed_world_size > 1
                and not self.args.use_bmuf
            ):
                self._wrapped_criterion = models.DistributedFairseqModel(
                    self.args, self._criterion
                )
            else:
                self._wrapped_criterion = self._criterion
        return self._wrapped_criterion

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.args.distributed_world_size > 1 and not self.args.use_bmuf:
                self._wrapped_model = models.DistributedFairseqModel(
                    self.args, self._model,
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler

    def _build_optimizer(self):
        params = list(
            filter(
                lambda p: p.requires_grad,
                chain(self.model.parameters(), self.criterion.parameters()),
            )
        )

        if self.args.fp16:
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                print('| WARNING: your device does NOT support faster training with --fp16, '
                      'please switch to FP32 which is likely to be faster')
            if self.args.memory_efficient_fp16:
                self._optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(self.args, params)
            else:
                self._optimizer = optim.FP16Optimizer.build_optimizer(self.args, params)
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                print('| NOTICE: your device may support faster training with --fp16')
            self._optimizer = optim.build_optimizer(self.args, params)

        if self.args.use_bmuf:
            self._optimizer = optim.FairseqBMUF(self.args, self._optimizer)

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)
        self._lr_scheduler.step_update(0)

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if distributed_utils.is_master(self.args):  # only save one checkpoint
            extra_state['train_meters'] = self.meters
            if self.data_actor is None:
                data_actor_state = None
                data_optimizer_state = None
            else:
                data_actor_state = self.data_actor.state_dict()
                data_optimizer_state = self.data_optimizer.state_dict()
            if self.extra_data_actor is None:
                extra_data_actor_state = None
                extra_data_optimizer_state = None
            else:
                extra_data_actor_state = self.extra_data_actor.state_dict()
                extra_data_optimizer_state = self.extra_data_optimizer.state_dict()


            checkpoint_utils.save_state(
                filename, self.args, self.get_model().state_dict(), self.get_criterion(),
                self.optimizer, self.lr_scheduler, self.get_num_updates(),
                self._optim_history, extra_state, data_actor_state, data_optimizer_state,
                extra_data_actor_state, extra_data_optimizer_state,
            )

    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
        data_actor_filename=None,
    ):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = None, [], None
        if self.args.only_load_data_actor:
            if os.path.exists(filename):
                state = checkpoint_utils.load_checkpoint_to_cpu(filename)
                if 'data_actor' in state and state['data_actor']:
                    print("loading data actor state only...")
                    self.data_actor.load_state_dict(state['data_actor'])
        else:
            if os.path.exists(filename):
                state = checkpoint_utils.load_checkpoint_to_cpu(filename)

                # load model parameters
                try:
                    self.get_model().load_state_dict(state['model'], strict=True)
                    if utils.has_parameters(self.get_criterion()):
                        self.get_criterion().load_state_dict(state['criterion'], strict=True)
                except Exception:
                    raise Exception(
                        'Cannot load model parameters from checkpoint {}; '
                        'please ensure that the architectures match.'.format(filename)
                    )

                extra_state = state['extra_state']
                self._optim_history = state['optimizer_history']
                last_optim_state = state.get('last_optimizer_state', None)

                if 'data_actor' in state and state['data_actor']:
                    self.data_actor.load_state_dict(state['data_actor'])
                if 'data_optimizer' in state and state['data_optimizer']:
                    self.data_optimizer.load_state_dict(state['data_optimizer'])
                if 'extra_data_actor' in state and state['extra_data_actor']:
                    self.extra_data_actor.load_state_dict(state['extra_data_actor'])
                if 'extra_data_optimizer' in state and state['extra_data_optimizer']:
                    self.extra_data_optimizer.load_state_dict(state['extra_data_optimizer'])

            if last_optim_state is not None and not reset_optimizer:
                # rebuild optimizer after loading model, since params may have changed
                self._build_optimizer()

                # only reload optimizer and lr_scheduler if they match
                last_optim = self._optim_history[-1]
                assert last_optim['criterion_name'] == self.get_criterion().__class__.__name__, \
                    'Criterion does not match; please reset the optimizer (--reset-optimizer).'
                assert last_optim['optimizer_name'] == self.optimizer.__class__.__name__, \
                    'Optimizer does not match; please reset the optimizer (--reset-optimizer).'

                if not reset_lr_scheduler:
                    self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
                self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

                self.set_num_updates(last_optim['num_updates'])

            if extra_state is not None:
                epoch = extra_state['train_iterator']['epoch']
                print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                    filename, epoch, self.get_num_updates()))

                self.lr_step(epoch)

                if 'train_meters' in extra_state and not reset_meters:
                    self.meters.update(extra_state['train_meters'])
                    del extra_state['train_meters']

                    # reset TimeMeters, since their start times don't make sense anymore
                    for meter in self.meters.values():
                        if isinstance(meter, TimeMeter):
                            meter.reset()
            else:
                print('| no existing checkpoint found {}'.format(filename))

        return extra_state

    def update_language_sampler_multilin(self, args):
        """Update the distribution to sample languages """
        # calculate gradient direction
        # calculate dev grad
        # Initialize dev data iterator
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        # #dev dataset x #train dataset
        all_sim_list = []
        valid_losses, train_losses = [], []
        for i, valid_key in enumerate(self.task.dataset('valid').datasets.keys()):
            #for _ in range(self.args.loss_steps):
            valid_sample = self.task.dataset('valid').get_sample_with_key(valid_key)
            valid_sample = self._prepare_sample(valid_sample)
            loss, sample_size, logging_output = self.task.train_step(
                                    valid_sample, self.model, self.criterion, self.optimizer)
            if sample_size > 0:
                loss = loss / sample_size
            valid_losses.append(loss)
            
            self.optimizer.save_dev_grad()
            self.zero_grad()
            if self.cuda:
                torch.cuda.empty_cache()
            sim_list = []
            for j, key in enumerate(self.task.dataset('train').datasets.keys()):
                if not args.no_dev and key == valid_key:
                    sim_list.append(1.0)
                else:
                    #for _ in range(self.args.loss_steps):
                    sample = self.task.dataset('train').get_sample_with_key(key)
                    sample = self._prepare_sample(sample)
                    # calculate sim
                    loss, sample_size, logging_output = self.task.train_step(
                                            sample, self.model, self.criterion, self.optimizer)
                    if sample_size > 0:
                        loss = loss / sample_size
                    train_losses.append(loss)
                    sim = self.optimizer.get_grad_sim()
                    sim_list.append(sim)
                    self.zero_grad()
                    if self.cuda:
                        torch.cuda.empty_cache()
            all_sim_list.append(sim_list)
        if args.pretrain_data_actor and not self.pretrained:
            if self.args.feature_type == 'ones':
                feature = torch.ones(1, len(self.task.dataset('train').datasets.keys()))
            elif self.args.feature_type == 'valid_loss':
                feature = torch.FloatTensor(valid_losses).view(1, -1)
                feature = feature/feature.sum()
            elif self.args.feature_type == 'train_loss':
                feature = torch.FloatTensor(train_losses).view(1, -1)
                feature = feature/feature.sum()
            else:
                print("feature not supported")
                exit(1)
            self.pretrained = True
            self.pretrain_data_actor(feature)
        # get rewards for languages based on different objectives
        if self.args.utility_type == 'ave':
            sim_list = np.mean(np.array(all_sim_list), axis=0).tolist()
            print(sim_list)
        elif self.args.utility_type == 'min-half':
            # find the valid languages with max losses
            # sort by loss, ascending order
            sorted_indices = np.argsort(valid_losses)
            selected_indices = sorted_indices[len(valid_losses)//2:]
            val_keys = list(self.task.dataset('valid').datasets.keys())
            print('selected keys:')
            for k in selected_indices:
                print(val_keys[k], valid_losses[k])
            selected_sim_list = []
            for k, sim in enumerate(all_sim_list):
                if k in selected_indices:
                    selected_sim_list.append(sim)
            sim_list = np.mean(np.array(selected_sim_list), axis=0).tolist()
            print(sim_list)
        elif self.args.utility_type == 'median':
            sorted_indices = np.argsort(valid_losses)
            selected_index = sorted_indices[len(valid_losses)//2]
            val_keys = list(self.task.dataset('valid').datasets.keys())
            print('selected keys:')
            print(val_keys[selected_index], valid_losses[selected_index])
            sim_list = all_sim_list[selected_index]
            print(sim_list)
        elif self.args.utility_type == 'ave_minus_weight':
            all_sim_list = np.array(all_sim_list)
            print(all_sim_list)
            sim_list = np.mean(all_sim_list, axis=0)
            print(self.task.dataset('train').p)
            current_p = np.array(self.task.dataset('train').p)
            current_p.resize(1, len(all_sim_list))
            weighted_sim = all_sim_list * current_p
            np.fill_diagonal(weighted_sim, 0)
            weighted_sim = np.sum(weighted_sim, axis=1)
            print(weighted_sim)
            sim_list = (sim_list - weighted_sim).tolist()
            print(sim_list)
        elif self.args.utility_type == 'ave_minus_baseline':
            all_sim_list = np.array(all_sim_list)
            print(all_sim_list)
            print(self.task.dataset('train').p)
            current_p = np.array(self.task.dataset('train').p)
            current_p.resize(1, len(all_sim_list))
            weighted_sim = all_sim_list * current_p
            weighted_sim = np.sum(weighted_sim, axis=1)
            print(weighted_sim)
            sim_list = np.mean((all_sim_list - weighted_sim), axis=0).tolist()
            print(sim_list)

        if self.args.baseline:
            cur_reward = np.array(sim_list)
            if self.baseline is None:
                self.baseline = np.mean(cur_reward)
            else:
                sim_list = cur_reward - self.baseline
                self.baseline = 0.5*self.baseline + 0.5*np.mean(cur_reward)
                sim_list = sim_list.tolist()
            print("baseline: {}".format(self.baseline))
            print("reward")
            print(sim_list)  
        if self.args.data_actor == 'base':
            if self.args.feature_type == 'ones':
                feature = torch.ones(1, len(self.task.dataset('train').datasets.keys()))
            elif self.args.feature_type == 'valid_loss':
                feature = torch.FloatTensor(valid_losses).view(1, -1)
                feature = feature/feature.sum()
            elif self.args.feature_type == 'train_loss':
                feature = torch.FloatTensor(train_losses).view(1, -1)
                feature = feature/feature.sum()
            print('feature')
            print(feature)
            grad_scale = torch.FloatTensor(sim_list).view(1, -1)

            if self.cuda:
                feature = feature.cuda()
                grad_scale = grad_scale.cuda()
            for _ in range(self.args.data_actor_optim_step):
                a_logits = self.data_actor.forward(feature)
                loss = -torch.nn.functional.log_softmax(a_logits, dim=-1)
                if self.args.scale_reward:
                    loss = loss * torch.softmax(a_logits, dim=-1).data
                loss = (loss * grad_scale).sum()
                loss.backward()
                self.data_optimizer.step()
                self.data_optimizer.zero_grad()
            with torch.no_grad():
                a_logits = self.data_actor.forward(feature)
                prob = torch.nn.functional.softmax(a_logits, dim=-1)
                sim_list = [i for i in prob.data.view(-1).cpu().numpy()]
        elif self.args.data_actor == 'base_weight':
            print(all_sim_list)
            feature = torch.ones(1, len(self.task.dataset('train').datasets.keys()))
            grad_scale = torch.FloatTensor(all_sim_list)
            if self.cuda:
                feature = feature.cuda()
                grad_scale = grad_scale.cuda()
            for _ in range(self.args.data_actor_optim_step):
                a_logits = self.data_actor.forward(feature)
                a_prob = torch.softmax(a_logits, dim=-1)
                loss = (grad_scale * a_prob.view(-1, 1)) * a_prob.view(1, -1)
                loss = loss.sum()
                loss.backward()
                self.data_optimizer.step()
                self.data_optimizer.zero_grad()
            with torch.no_grad():
                a_logits = self.data_actor.forward(feature)
                prob = torch.nn.functional.softmax(a_logits, dim=-1)
                sim_list = [i for i in prob.data.view(-1).cpu().numpy()]
        elif self.args.data_actor == 'only_grad':
            sim_list = np.exp(sim_list)
            sim_list = sim_list/np.sum(sim_list)
        elif self.args.data_actor == 'interpolate_grad':
            orig_p = np.array(self.task.dataset('train').p)
            log_p = np.log(orig_p)
            print(log_p)
            log_p += np.array(sim_list) * self.args.data_actor_lr
            print(log_p)
            sim_list = np.exp(log_p)
        # set sampling distribution
        self.task.dataset('train').update_sampling_distribution(sim_list)


    def update_language_sampler(self, args):
        """Update the distribution to sample languages """
        # calculate gradient direction
        # calculate dev grad
        # Initialize dev data iterator
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        if not args.no_dev:
            dev_itr = self.task.get_batch_iterator(
                dataset=self.task.dataset('valid'),
                max_tokens=args.max_tokens_valid,
                max_sentences=args.max_sentences_valid,
                max_positions=utils.resolve_max_positions(
                    self.task.max_positions(),
                    self.get_model().max_positions(),
                ),
                ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
                required_batch_size_multiple=args.required_batch_size_multiple,
                seed=args.seed,
                num_shards=args.distributed_world_size,
                shard_id=args.distributed_rank,
                num_workers=args.num_workers,
            ).next_epoch_itr(shuffle=True)
            for sample in dev_itr:
                sample = self._prepare_sample(sample)
                assert sample is not None
                loss, sample_size, logging_output = self.task.train_step(
                                        sample, self.model, self.criterion, self.optimizer)
                self.optimizer.save_dev_grad()
                #self.optimizer.zero_grad()
                break
            self.zero_grad()
        sim_list = []
        for i, key in enumerate(self.task.dataset('train').datasets.keys()):
            sample = self.task.dataset('train').get_sample_with_key(key)
            sample = self._prepare_sample(sample)
            # calculate sim
            loss, sample_size, logging_output = self.task.train_step(
                                    sample, self.model, self.criterion, self.optimizer)
            if args.no_dev and i == 0:
                self.optimizer.save_dev_grad()
            sim = self.optimizer.get_grad_sim()
            sim_list.append(sim)
            self.zero_grad()

        if self.args.data_actor == 'base':
            feature = torch.ones(1, len(self.task.dataset('train').datasets.keys()))
            grad_scale = torch.FloatTensor(sim_list).view(1, -1)
            if self.cuda:
                feature = feature.cuda()
                grad_scale = grad_scale.cuda()
            for _ in range(self.args.data_actor_optim_step):
                a_logits = self.data_actor.forward(feature)
                loss = -torch.nn.functional.log_softmax(a_logits, dim=-1)
                loss = (loss * grad_scale).sum()
                loss.backward()
                self.data_optimizer.step()
                self.data_optimizer.zero_grad()
            with torch.no_grad():
                a_logits = self.data_actor.forward(feature)
                prob = torch.nn.functional.softmax(a_logits, dim=-1)
                sim_list = [i for i in prob.data.view(-1).cpu().numpy()]
        # set sampling distribution
        self.task.dataset('train').update_sampling_distribution(sim_list)
    
    def pretrain_data_actor(self, feature):
        """pretrain the distribution to sample languages """
        if self.args.data_actor == 'base' or self.args.data_actor == 'base_weight':
            if self.args.pretrain_type == "lan_dist":
                target = torch.FloatTensor(args.lan_dists).view(1, -1)
            elif self.args.pretrain_type == "datasize":
                datasize_p = self.task.dataset('train').p
                target = torch.FloatTensor(datasize_p).view(1, -1)
            print(target)
            for p in self.data_optimizer.param_groups:
                p['lr'] = 0.001
            
            if self.cuda:
                feature = feature.cuda()
                target = target.cuda()
            l = 100
            while l > 0.000001:
                a_logits = self.data_actor.forward(feature)
                prob = torch.nn.functional.softmax(a_logits, dim=-1)
                loss = torch.nn.functional.mse_loss(prob, target)
                l = loss.item()
                loss.backward()
                self.data_optimizer.step()
                self.data_optimizer.zero_grad()
            with torch.no_grad():
                a_logits = self.data_actor.forward(feature)
                prob = torch.nn.functional.softmax(a_logits, dim=-1)
                sim_list = [i for i in prob.data.view(-1).cpu().numpy()]
                print("pretrained_sim", sim_list)

            for p in self.data_optimizer.param_groups:
                p['lr'] = self.args.data_actor_lr

    def get_train_iterator(self, epoch, combine=True):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        print('| loading train data for epoch {}'.format(epoch))
        self.task.load_dataset(self.args.train_subset, epoch=epoch, combine=combine)
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(self.args.train_subset),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            seed=self.args.seed,
            num_shards=self.args.distributed_world_size,
            shard_id=self.args.distributed_rank,
            num_workers=self.args.num_workers,
            epoch=epoch,
        )

    def train_step(self, samples, dummy_batch=False, raise_oom=False, update_actor=True):
        """Do forward, backward and parameter update."""
        if self._dummy_batch is None:
            self._dummy_batch = samples[0]

        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        if not dummy_batch:
            self.meters['train_wall'].start()

        # forward and backward pass
        logging_outputs, sample_sizes, ooms = [], [], 0
        for i, sample in enumerate(samples):
            sample = self._prepare_sample(sample)
            if sample is None:
                # when sample is None, run forward/backward on a dummy batch
                # and ignore the resulting gradients
                sample = self._prepare_sample(self._dummy_batch)
                ignore_grad = True
            else:
                ignore_grad = False

            def maybe_no_sync():
                """
                Whenever *samples* contains more than one mini-batch, we
                want to accumulate gradients locally and only call
                all-reduce in the last backwards pass.
                """
                if (
                    self.args.distributed_world_size > 1
                    and hasattr(self.model, 'no_sync')
                    and i < len(samples) - 1
                ):
                    return self.model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            try:
                with maybe_no_sync():
                    # forward and backward
                    if self.args.data_actor == 'ave_emb' and update_actor:
                        self.optimizer.clone_param()
                        data_actor = self.data_actor
                        cached_loss = {}
                        data_actor_out = {}
                    elif self.args.extra_data_actor == 'ave_emb' and update_actor:
                        self.optimizer.clone_param()
                        data_actor = self.extra_data_actor
                        cached_loss = {}
                        data_actor_out = {}
                    else:
                        data_actor = None
                        cached_loss = None
                        data_actor_out = None
                    loss, sample_size, logging_output = self.task.train_step(
                        sample, self.model, self.criterion, self.optimizer,
                        ignore_grad, data_actor=data_actor, 
                        loss_copy=cached_loss, data_actor_out=data_actor_out,
                    )
                if not ignore_grad:
                    logging_outputs.append(logging_output)
                    sample_sizes.append(sample_size)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    msg = (
                        '| WARNING: ran out of memory with exception: '
                        + '{};'.format(e)
                        + '\n Skipping batch'
                    )
                    # TODO: print should really go to logger, this print goes
                    # to stdout, which is buffered, which in many case is not
                    # printed out if another exception happens
                    # print(msg)
                    print(msg, file=sys.stderr)
                    if raise_oom:
                        raise ValueError(msg)
                    ooms += 1
                    self.zero_grad()
                else:
                    raise e

        if ooms > 0 and self._oom_batch is not None:
            self.handle_ooms(ooms)

        if dummy_batch:
            return None

        # gather logging outputs from all replicas
        if self.args.distributed_world_size > 1 and (
            (not self.args.use_bmuf)
            or (
                self.args.use_bmuf
                and (self.get_num_updates() + 1) % self.args.global_sync_iter == 0
            )
        ):
            logging_outputs, sample_sizes, ooms, prev_norms = \
                zip(*distributed_utils.all_gather_list(
                    [logging_outputs, sample_sizes, ooms, self._prev_grad_norm],
                ))
            logging_outputs = list(chain.from_iterable(logging_outputs))
            sample_sizes = list(chain.from_iterable(sample_sizes))
            ooms = sum(ooms)

            if not self.args.use_bmuf:
                assert (
                    all(norm == prev_norms[0] for norm in prev_norms)
                    or all(math.isnan(norm) or math.isinf(norm) for norm in prev_norms)
                ), 'Fatal error: gradients are inconsistent between workers'

        self.meters['oom'].update(ooms, len(samples))
        if ooms == self.args.distributed_world_size * len(samples):
            print('| WARNING: OOM in all workers, skipping update')
            self.zero_grad()
            return None

        # aggregate logging outputs and sample sizes
        logging_output = self.task.aggregate_logging_outputs(
            logging_outputs, self.get_criterion()
        )
        sample_size = self.task.grad_denom(sample_sizes, self.get_criterion())
        #print(logging_output)
        #print(logging_outputs)
        if not all(k in logging_output for k in ['ntokens', 'nsentences']):
            raise Exception((
                'Please update the {}.aggregate_logging_outputs() method to '
                'return ntokens and nsentences'
            ).format(self.task.__class__.__name__))

        try:
            # normalize grads by sample size
            if sample_size > 0:
                self.optimizer.multiply_grads(self.args.distributed_world_size / float(sample_size))

            # clip grads
            grad_norm = self.optimizer.clip_grad_norm(self.args.clip_norm)
            self._prev_grad_norm = grad_norm

            # take an optimization step
            self.optimizer.step()
            self.set_num_updates(self.get_num_updates() + 1)

            # task specific update per step
            self.task.update_step(self._num_updates)

            # update meters
            ntokens = logging_output.get('ntokens', 0)
            nsentences = logging_output.get('nsentences', 0)
            self.meters['wps'].update(ntokens)
            self.meters['ups'].update(1.)
            self.meters['wpb'].update(ntokens)
            self.meters['bsz'].update(nsentences)
            self.meters['gnorm'].update(grad_norm)
            self.meters['clip'].update(
                1. if grad_norm > self.args.clip_norm and self.args.clip_norm > 0 else 0.
            )
            self.meters['train_loss'].update(logging_output.get('loss', 0), sample_size)
            if 'train_acc' in self.meters:
                self.meters['train_acc'].update(
                    logging_output.get('acc', 0), sample_size)

            if 'nll_loss' in logging_output:
                self.meters['train_nll_loss'].update(logging_output.get('nll_loss', 0), ntokens)
        except OverflowError as e:
            print('| WARNING: overflow detected, ' + str(e))
            self.zero_grad()
            logging_output = None

        if self.args.fp16:
            self.meters['loss_scale'].reset()
            self.meters['loss_scale'].update(self.optimizer.scaler.loss_scale)

        self.meters['train_wall'].stop()

        if (self.args.data_actor == 'ave_emb' or self.args.extra_data_actor == 'ave_emb') and update_actor:
            # update data actor
            # get dev gradient
            if self.args.data_actor_multilin:
                for i, valid_key in enumerate(self.task.dataset('valid').datasets.keys()):
                    valid_sample = self.task.dataset('valid').get_sample_with_key(valid_key)
                    valid_sample = self._prepare_sample(valid_sample)
                    _loss, _sample_size, _logging_output = self.task.train_step(
                                            valid_sample, self.model, self.criterion, self.optimizer)
                    if self.args.utility_type == 'ave':
                        extras = (i==0)
                    else:
                        print("not supported yet")
                        exit(0)
                    self.optimizer.save_dev_grad_multi(utility=self.args.data_actor_multilin, extras=extras)
                    self.zero_grad()
                    if self.cuda:
                        torch.cuda.empty_cache()
                if self.args.utility_type == 'ave':
                    extras = len(self.task.dataset('valid').datasets.keys())
                self.optimizer.multi_dev_grad_finalize(utility=self.args.data_actor_multilin, extras=extras)
            else:
                dev_itr = self.task.get_batch_iterator(
                    dataset=self.task.dataset('valid'),
                    max_tokens=self.args.max_tokens_valid,
                    max_sentences=self.args.max_sentences_valid,
                    max_positions=utils.resolve_max_positions(
                        self.task.max_positions(),
                        self.get_model().max_positions(),
                    ),
                    ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
                    required_batch_size_multiple=self.args.required_batch_size_multiple,
                    seed=self.args.seed,
                    num_shards=self.args.distributed_world_size,
                    shard_id=self.args.distributed_rank,
                    num_workers=self.args.num_workers,
                ).next_epoch_itr(shuffle=True)
                for valid_sample in dev_itr:
                    valid_sample = self._prepare_sample(valid_sample)
                    _loss, _sample_size, _logging_output = self.task.train_step(
                                            valid_sample, self.model, self.criterion, self.optimizer)
                    self.optimizer.save_dev_grad()
                    break
                self.zero_grad()

            # get per example reward
            with torch.no_grad():
                self.optimizer.switch_param()
                eta = 0.001
                self.optimizer.add_grad(eta=eta)
                cur_loss = {}
                _loss, _sample_size, _logging_output = self.task.train_step(
                    sample, self.model, self.criterion, self.optimizer,
                    ignore_grad=True, data_actor=None, loss_copy=cur_loss,
                )
                self.optimizer.switch_param(clear_cache=True)
            # optimize data actor
            for k in cached_loss.keys():
                reward = 1./eta * (cur_loss[k] - cached_loss[k])
                if self.args.out_score_type == 'sigmoid':
                    #loss = -(torch.log(1e-20 + data_actor_out[k]) * reward.data)
                    loss = -(data_actor_out[k] * reward.data)
                elif self.args.out_score_type == 'exp':
                    loss = -(torch.log(1e-20 + data_actor_out[k]) * reward.data)
                if cur_loss[k].size(0) > 0:
                    loss.div_(cur_loss[k].size(0))
                loss.sum().backward()
            if self.args.data_actor == 'ave_emb': 
                self.data_optimizer.step()
                self.data_optimizer.zero_grad()
            elif self.args.extra_data_actor == 'ave_emb':
                self.extra_data_optimizer.step()
                self.extra_data_optimizer.zero_grad()

        return logging_output

    def valid_step(self, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            sample = self._prepare_sample(sample)
            if sample is None:
                sample = self._prepare_sample(self._dummy_batch)
                ignore_results = True
            else:
                ignore_results = False

            try:
                _loss, sample_size, logging_output = self.task.valid_step(
                    sample, self.model, self.criterion
                )
            except RuntimeError as e:
                if 'out of memory' in str(e) and not raise_oom:
                    print('| WARNING: ran out of memory, retrying batch')
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad = None  # free some memory
                    if self.cuda:
                        torch.cuda.empty_cache()
                    return self.valid_step(sample, raise_oom=True)
                else:
                    raise e

            if ignore_results:
                logging_output, sample_size = {}, 0

        # gather logging outputs from all replicas
        if self.args.distributed_world_size > 1:
            logging_output, sample_size = zip(*distributed_utils.all_gather_list(
                [logging_output, sample_size],
            ))
            logging_output = list(logging_output)
            sample_size = list(sample_size)
        else:
            logging_output = [logging_output]
            sample_size = [sample_size]

        # aggregate logging outputs and sample sizes
        logging_output = self.task.aggregate_logging_outputs(
            logging_output, self.get_criterion()
        )
        sample_size = self.task.grad_denom(
            sample_size, self.get_criterion()
        )
        # update meters for validation
        ntokens = logging_output.get('ntokens', 0)
        self.meters['valid_loss'].update(logging_output.get('loss', 0), sample_size)
        if 'valid_acc' in self.meters:
            self.meters['valid_acc'].update(
                logging_output.get('acc', 0), sample_size)

        if 'nll_loss' in logging_output:
            self.meters['valid_nll_loss'].update(logging_output.get('nll_loss', 0), ntokens)

        return logging_output

    def dummy_train_step(self, dummy_batch):
        """Dummy training step for warming caching allocator."""
        self.train_step(dummy_batch, dummy_batch=True)
        self.zero_grad()

    def handle_ooms(self, number_of_ooms):
        """
        c10d accumulates/syncs gradients between gpus during backward pass.
        In case of OOMs, gpus may fail to sync, so we manually iterate
        extra to make sure each gpu makes same number of iterations.
        """
        for _ in range(number_of_ooms):
            self.train_step([self._oom_batch], True)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        self.lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step_update(self):
        """Update the learning rate after each update."""
        return self.lr_scheduler.step_update(self.get_num_updates())

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_criterion(self):
        """Get the (non-wrapped) criterion instance."""
        return self._criterion

    def get_meter(self, name):
        """Get a specific meter by name."""
        if name not in self.meters:
            return None
        return self.meters[name]

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None

        if self.cuda:
            sample = utils.move_to_cuda(sample)

        def apply_half(t):
            if t.dtype is torch.float32:
                return t.half()
            return t

        if self.args.fp16:
            sample = utils.apply_to_sample(apply_half, sample)

        return sample

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed + self.get_num_updates()
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.manual_seed(seed)
