# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import utils
import torch
from . import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True, data_score=None, val_loss_data=None, loss_copy=False, args=None):
    B, T = target.size(0), target.size(1)
    target = target.view(-1, 1)
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    if loss_copy:
        # keep a copy of nll_loss data
        # lprobs: [BT X VSIZE]
        #nll_loss_data = -lprobs.gather(dim=-1, index=target).data.clone()
        nll_loss_data = -lprobs.gather(dim=-1, index=target)
        # nll_loss_data: [BT X 1]
        if ignore_index is not None:
            pad_mask = target.eq(ignore_index)
            nll_loss_data.masked_fill_(pad_mask, 0.)
        nll_loss_data = nll_loss_data.view(B, T)
    else:
        nll_loss_data = None

    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if val_loss_data is not None:
        if ignore_index is not None:
            pad_mask = target.eq(ignore_index)
            val_loss_data.masked_fill_(pad_mask, 0.)
            nll_loss_data.masked_fill_(pad_mask, 0.)
            tgt_len = (~pad_mask).float().view(B, -1).sum(dim=1, keepdim=True)
        else:
            tgt_len = T + 0.0
        dev_grad_dotprod = (val_loss_data - nll_loss_data).view(B, -1).sum(dim=1, keepdim=True).data / tgt_len
        #dev_grad_dotprod = (val_loss_data - nll_loss_data).view(B, -1).data
        #if args.reward_level == 'sent':
        #    reward = (data_score - nll_loss_data).view(B, -1)
        #    # v8
        #    #reward = reward.sum(dim=1) * 0.04
        #    reward = reward.sum(dim=1) * args.reward_constant
        #    reward = torch.nn.functional.softmax(reward, dim=0).unsqueeze(1) * B
        #elif args.reward_level == 'word':
        #    reward = (data_score - nll_loss_data).masked_fill_(pad_mask, -float("inf")) * args.reward_constant
        #    if ignore_index is not None:
        #        nwords = (~pad_mask).long().sum()
        #    else:
        #        nwords = B*T
        #    reward = torch.nn.functional.softmax(reward, dim=0) * nwords
        #    reward = reward.view(B, -1)
        ##print(reward)
        #lprobs = (lprobs.view(B, -1, lprobs.size(-1))*reward.data.unsqueeze(2)).view(-1, lprobs.size(-1))
        ##lprobs = (lprobs.view(data_score.size(0), -1, lprobs.size(-1))*data_score.data.unsqueeze(2)).view(-1, lprobs.size(-1))
    else:
        dev_grad_dotprod = None

    nll_loss = -lprobs.gather(dim=-1, index=target)
    if data_score is not None:
        if args.relu_reward:
            reward = torch.nn.functional.relu(data_score.data)
        else:
            reward = data_score.data
        if args.discount_reward > 0:
            discount = [1]
            for i in range(1, T):
                discount.append(discount[-1] * args.discount_reward)
            discount.reverse()
            discount = torch.FloatTensor([discount])
            if reward.is_cuda:
                discount = discount.cuda()
            reward = reward.repeat(1, T) * discount
        #print(reward)
        nll_loss = nll_loss.view(B, -1) * reward
        nll_loss = nll_loss.view(-1, 1)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss, nll_loss_data, dev_grad_dotprod


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.args = args

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True, val_loss_data=None, data_score=None, loss_copy=False, debug_print=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, nll_loss_data, dev_grad_dotprod = self.compute_loss(model, net_output, sample, reduce=reduce, val_loss_data=val_loss_data, data_score=data_score, loss_copy=loss_copy, debug_print=debug_print)
        if debug_print:
            print(nll_loss_data)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output, nll_loss_data, dev_grad_dotprod

    def compute_loss(self, model, net_output, sample, reduce=True, data_score=None, val_loss_data=None, loss_copy=False, debug_print=False):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)
        if debug_print:
            print(target)
        loss, nll_loss, nll_loss_data, dev_grad_dotprod = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce, data_score=data_score, val_loss_data=val_loss_data, loss_copy=loss_copy, args=self.args, 
        )
        return loss, nll_loss, nll_loss_data, dev_grad_dotprod

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
