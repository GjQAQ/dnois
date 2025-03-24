import torch

from .base.typing import Ts
from . import torch as _t

__all__ = [
    'center_depth',
    'depth2ips',
    'depth2slope',
    'depth_range',
    'ips2depth',
    'ips2slope',
    'quantize_depth_map',
    'slope2depth',
    'slope2ips',
    'slope_lim',
    'slope_range'
]


def depth2ips(depth, min_depth, max_depth):
    return max_depth * (depth - min_depth) / (depth * (max_depth - min_depth))


def ips2depth(ips, min_depth, max_depth):
    return max_depth * min_depth / (max_depth - ips * (max_depth - min_depth))


def depth2slope(depth, central_depth):
    return (depth - central_depth) / depth


def slope2depth(slope, central_depth):
    return central_depth / (1 - slope)


def ips2slope(ips, slope_range_):
    return slope_range_ * (ips - 0.5)


def slope2ips(slope, slope_range_):
    return slope / slope_range_ + 0.5


def center_depth(min_depth, max_depth):
    return 2 * min_depth * max_depth / (min_depth + max_depth)


def slope_lim(min_depth, max_depth):
    return (max_depth - min_depth) / (max_depth + min_depth)


def slope_range(min_depth, max_depth):
    return 2 * slope_lim(min_depth, max_depth)


def depth_range(central_depth, slope_range_):
    _center_depth2 = 2 * central_depth
    return _center_depth2 / (2 + slope_range_), _center_depth2 / (2 - slope_range_)


def quantize_depth_map(dmap: Ts, min_depth, max_depth, n: int, binary: bool = False, eps: float = 1e-8) -> Ts:
    """
    :param Tensor dmap: A tensor of any shape ``(...)``.
    :param min_depth: Minimum valid depth.
    :type min_depth: float or Tensor
    :param max_depth: Maximum valid depth.
    :type max_depth: float or Tensor
    :param int n: Number of slices i.e. quantization level.
    :param bool binary: Whether to return a binary mask. Default: ``False``.
    :param float eps: A small number to ensure numerical stability. Default: ``1e-8``.
    :return: A tensor of shape ``(n, ...)`` where the first dimension means different slices.
    """
    dmap = dmap.clamp(min_depth, max_depth)
    imap = depth2ips(dmap, min_depth, max_depth)
    imap = imap.clamp(eps, 1) * n
    ips = torch.arange(1, n + 1, dtype=imap.dtype, device=imap.device)  # ...
    ips = _t.as1d(ips, imap.ndim + 1, 0)  # N x ...
    diff = ips - imap  # N x ...

    if binary:
        alpha = torch.zeros_like(diff, dtype=torch.bool)
        alpha[diff.ge(0) & diff.lt(1)] = 1
    else:
        alpha = torch.zeros_like(diff)
        mask = diff.gt(-1) & diff.le(0)
        alpha[mask] = diff[mask] + 1
        alpha[diff.gt(0) & diff.le(1)] = 1
    return alpha
