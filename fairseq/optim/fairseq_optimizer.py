# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch


class FairseqOptimizer(object):

    def __init__(self, args):
        super().__init__()
        self.args = args

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        pass

    @property
    def optimizer(self):
        """Return a torch.optim.optimizer.Optimizer instance."""
        if not hasattr(self, '_optimizer'):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError('_optimizer must be an instance of torch.optim.Optimizer')
        return self._optimizer

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        raise NotImplementedError

    @property
    def params(self):
        """Return an iterable of the parameters held by the optimizer."""
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                yield p

    def save_dev_grad_multi(self, utility='ave', extras=None):
        """Save dev set gradient"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                if utility == 'ave':
                    if extras == True:
                        state['dev_grad'] = p.grad.data.clone()
                    else:
                        state['dev_grad'] += p.grad.data.clone()

    def multi_dev_grad_finalize(self, utility='ave', extras=None):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                if utility == 'ave':
                    state['dev_grad'].div_(extras)
    
    def save_train_grad_id(self, i):
        """Save train set gradient"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                if 'train_grad' not in state:
                    state['train_grad'] = [None for _ in range(len(self.args.lang_pairs))]
                if state['train_grad'][i] is None:
                    state['train_grad'][i] = p.grad.data.clone()
                else:
                    #state['train_grad'][i] = p.grad.data.clone()
                    state['train_grad'][i] = self.args.a1*p.grad.data + self.args.a0*state['train_grad'][i]

    def get_grad_sim_id(self, i):
        """Get gradient similarity with dev set gradient"""
        cosine_prod, cosine_norm, dev_cosine_norm = 0, 0, 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                cosine_prod += (state['train_grad'][i] * p.grad.data).sum().item()
                cosine_norm += p.grad.data.norm(2) ** 2
                dev_cosine_norm += state['train_grad'][i].norm(2) ** 2
        if self.args.grad_sim == "cosine":
            cosine_sim = cosine_prod / ((cosine_norm*dev_cosine_norm)**0.5 + 1e-10)
            return cosine_sim.item(), cosine_norm, dev_cosine_norm
        elif self.args.grad_sim == "dot_prod":
            cosine_sim = cosine_prod 
            return cosine_sim, cosine_norm, dev_cosine_norm

    def save_train_grad(self):
        """Save train set gradient"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                state['dev_grad'] = p.grad.data.clone() - state['dev_grad']

    def save_train_grad_t0(self):
        """Save train set gradient"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                state['dev_grad'] = p.grad.data.clone()

    def save_dev_grad(self):
        """Save dev set gradient"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                state['dev_grad'] = p.grad.data.clone()

    def clone_param(self):
        """Save a copy of the params"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state[p]
                state['param_copy'] = p.data.clone()

    def add_grad(self, eta):
        """add grad to current param"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                p.data += state['dev_grad']*eta

    def switch_param(self, clear_cache=False):
        """Swap copy and the param values"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state[p]
                cur_p = p.data
                p.data = state['param_copy']
                if clear_cache:
                    state['param_copy'] = None 
                else:
                    state['param_copy'] = cur_p

    def get_grad_sim(self):
        """Get gradient similarity with dev set gradient"""
        cosine_prod, cosine_norm, dev_cosine_norm = 0, 0, 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.optimizer.state[p]
                cosine_prod += (state['dev_grad'] * p.grad.data).sum().item()
                cosine_norm += p.grad.data.norm(2) ** 2
                dev_cosine_norm += state['dev_grad'].norm(2) ** 2
        if self.args.grad_sim == "cosine":
            cosine_sim = cosine_prod / ((cosine_norm*dev_cosine_norm)**0.5 + 1e-10)
            return cosine_sim.item(), cosine_norm, dev_cosine_norm
        elif self.args.grad_sim == "dot_prod":
            cosine_sim = cosine_prod 
            return cosine_sim, cosine_norm, dev_cosine_norm

    def __getstate__(self):
        return self._optimizer.__getstate__()

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        self.optimizer.load_state_dict(state_dict)

        if optimizer_overrides is not None and len(optimizer_overrides) > 0:
            # override learning rate, momentum, etc. with latest values
            for group in self.optimizer.param_groups:
                group.update(optimizer_overrides)

    def backward(self, loss, retain_graph=False):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves."""
        loss.backward(retain_graph=retain_graph)

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for p in self.params:
            if p.grad is not None:
                p.grad.data.mul_(c)

    def clip_grad_norm(self, max_norm):
        """Clips gradient norm."""
        if max_norm > 0:
            return torch.nn.utils.clip_grad_norm_(self.params, max_norm)
        else:
            return math.sqrt(sum(p.grad.data.norm()**2 for p in self.params if p.grad is not None))

    def step(self, closure=None):
        """Performs a single optimization step."""
        self.optimizer.step(closure)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.params:
            p.grad = None
        self.optimizer.zero_grad()

    @property
    def supports_memory_efficient_fp16(self):
        if hasattr(self.optimizer, 'supports_memory_efficient_fp16'):
            return self.optimizer.supports_memory_efficient_fp16
        return False
