import torch

from ..base.typing import Ts

__all__ = [
    'gaussian',
    'poisson',
]


def gaussian(signal: Ts, sigma: float | Ts, noise_only: bool = False) -> Ts:
    r"""
    Applying gaussian noise to signal:

    .. math::
        \tilde{\mathbf{x}}=\mathbf{x}+\mathbf{n}, \mathbf{n}\sim\mathcal{N}(0,\sigma^2\mathbf{I}).

    :param Tensor signal: Input signal.
    :param sigma: Standard deviation of the gaussian noise.
        Must be broadcastable with ``signal`` if a Tensor.
    :type sigma: float | Tensor
    :param bool noise_only: If ``True``, return noise rather than noisy signal. Default: ``False``.
    :return: Noisy signal.
    :rtype: Tensor
    """
    noise = sigma * torch.randn_like(signal)
    return noise if noise_only else signal + noise


def poisson(signal: Ts, a: float | Ts, noise_only: bool = False) -> Ts:
    r"""
    Applying poissonian noise to signal:

    .. math::
        \tilde{\mathbf{x}}\sim\mathcal{P}(\mathbf{x}/a).

    :param Tensor signal: Input signal.
    :param a: Scale factor :math:`a` in Poisson distribution.
        Must be broadcastable with ``signal`` if a Tensor.
    :type a: float | Tensor
    :param bool noise_only: If ``True``, return noise rather than noisy signal. Default: ``False``.
    :return: Noisy signal.
    :rtype: Tensor
    """
    noisy = torch.poisson(signal / a)
    return torch.detach(noisy - signal) + signal if noise_only else noisy
