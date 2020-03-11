# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.optim

from . import FairseqOptimizer, register_optimizer


@register_optimizer('data_sgd')
class DATASGD(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = torch.optim.SGD(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--bt-optimizer-momentum', default=0.0, type=float,
                            help='momentum factor')
        parser.add_argument('--bt-optimizer-weight-decay', default=0.0, type=float,
                            help='weight decay')
        parser.add_argument('--bt-optimizer-nesterov', action="store_true")
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.data_actor_lr[0],
            'momentum': self.args.bt_optimizer_momentum,
            'weight_decay': self.args.bt_optimizer_weight_decay,
            'nesterov': self.args.bt_optimizer_nesterov,
        }
