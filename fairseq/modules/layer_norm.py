# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

class ScaleNorm(torch.nn.Module):
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x*norm

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False, scale=None):
    #if not export and torch.cuda.is_available():
    #    try:
    #        from apex.normalization import FusedLayerNorm
    #        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    #    except ImportError:
    #        pass
    if scale is not None:
        return ScaleNorm(scale)
    else:
        return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
