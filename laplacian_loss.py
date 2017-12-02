from __future__ import division, print_function
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

"""
The origin author :mtyka
source : https://github.com/mtyka/laploss
"""


def gauss_kernel(size=5, sigma=1):
    grid = np.mgrid[:size, :size].T
    center = [size // 2] * 2
    kernel = np.exp(((grid - center) ** 2).sum(axis=2) / (-2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


_gauss_kernel_weights = {}


def conv_gauss_kernel(x, k_size=5, sigma=1, stride=1, cuda=False,
                      padding='same'):
    assert k_size % 2 == 1
    n_channels = x.size()[1]

    global _gauss_kernel_weights
    idx = (k_size, sigma, cuda, n_channels)
    if idx in _gauss_kernel_weights:
        weights = _gauss_kernel_weights[idx]
    else:
        weights = torch.from_numpy(gauss_kernel(size=k_size, sigma=sigma))
        if cuda:
            weights = weights.cuda()
        weights = Variable(weights, requires_grad=False)
        weights = weights.expand(n_channels, 1, k_size, k_size)
        if cuda:
            weights = weights.contiguous()
        _gauss_kernel_weights[idx] = weights

    padding_amt = k_size // 2
    padding_kw = {}
    if padding == 'reflect':
        x = torch.nn.ReflectionPad2d(padding_amt)(x)
        # NOTE: breaks if the pyramid gets too small:
        #       https://github.com/pytorch/pytorch/issues/2563
    elif padding in {'same', 'replicate'}:
        x = torch.nn.ReplicationPad2d(padding_amt)(x)
    elif padding == 'zero':
        padding_kw['padding'] = padding_amt
    else:
        raise ValueError("unknown padding type {}".format(padding))

    return F.conv2d(x, weights, stride=stride, groups=n_channels, **padding_kw)


def laplacian_pyramid(x, n_levels=-1, k_size=9, sigma=2, padding='same',
                      cuda=False, downscale=2):
    if n_levels == -1:  # as many levels as possible
        n_levels = int(np.ceil(math.log(max(x.size()[-2:]), downscale)))

    pyr = []
    current = x
    for level in range(n_levels):
        gauss = conv_gauss_kernel(
            current, k_size=k_size, sigma=sigma, padding=padding, cuda=cuda)
        diff = current - gauss
        pyr.append(diff)
        current = F.avg_pool2d(gauss, downscale)
    pyr.append(current)
    return pyr


def laplacian_loss(input, target, **kwargs):
    pyr_i = laplacian_pyramid(input, **kwargs)
    pyr_t = laplacian_pyramid(target, **kwargs)
    loss = 0
    for j, (i, t) in enumerate(zip(pyr_i, pyr_t),1):
        wt = 2 ** (-2 * j)
        loss += wt * torch.mean(torch.abs(i-t))
    return loss
