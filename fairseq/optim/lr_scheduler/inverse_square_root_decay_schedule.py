# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import FairseqLRScheduler, register_lr_scheduler
import torch

@register_lr_scheduler('inverse_sqrt_decay')
class InverseSquareRootDecaySchedule(FairseqLRScheduler):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with inverse_sqrt.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        warmup_end_lr = args.lr[0]
        if args.warmup_init_lr < 0:
            args.warmup_init_lr = 0 if args.warmup_updates > 0 else warmup_end_lr

        # linearly warmup for the first args.warmup_updates
        self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * args.warmup_updates**0.5

        # initial learning rate
        self.lr = args.warmup_init_lr
        self.optimizer.set_lr(self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer.optimizer, patience=0, factor=args.lr_shrink,
            threshold=args.lr_threshold)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--warmup-updates', default=4000, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                            help='initial learning rate during warmup phase; default is args.lr')
        parser.add_argument('--lr-shrink', default=0.1, type=float, metavar='LS',
                            help='shrink factor for annealing, lr_new = (lr * lr_shrink)')
        parser.add_argument('--lr-threshold', default=1e-4, type=float, metavar='LT',
                            help='Threshold for measuring the new optimum, \
                            to only focus on significant changes')
        # fmt: on

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        if val_loss is not None:
            self.lr_scheduler.step(val_loss, epoch)
        else:
            self.lr_scheduler.last_epoch = epoch
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.args.warmup_updates:
            self.lr = self.args.warmup_init_lr + num_updates*self.lr_step
        else:
            self.lr = self.decay_factor * num_updates**-0.5
        self.optimizer.set_lr(self.lr)
        return self.lr
