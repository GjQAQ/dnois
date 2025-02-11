import torch

from ..base.typing import Ts, Sequence

__all__ = [
    'gaussian',
    'poisson',
]


def gaussian(
    signal: Ts,
    sigma: float | tuple[float, float] | Ts,
    noise_only: bool = False,
    dims: int | Sequence[int] = 0,
) -> Ts:
    r"""
    Applying gaussian noise to signal:

    .. math::
        \tilde{\mathbf{x}}=\mathbf{x}+\mathbf{n}, \mathbf{n}\sim\mathcal{N}(0,\sigma^2\mathbf{I}).

    :param Tensor signal: Input signal.
    :param sigma: Standard deviation (std) of the gaussian noise.
        Must be broadcastable with ``signal`` if a Tensor.
        If a 2-tuple of float ``(min, max)``, the resulted std will vary along dimensions ``dims``
        and follows a uniform distribution in this range.
    :type sigma: float | tuple[float, float] | Tensor
    :param bool noise_only: If ``True``, return noise rather than noisy signal. Default: ``False``.
    :param dims: Dimensions with varied standard deviation. See description for ``sigma``. Default: ``0``.
    :type dims: int | Sequence[int]
    :return: Noisy signal.
    :rtype: Tensor
    """
    noise = torch.randn_like(signal)
    if isinstance(sigma, tuple):
        if len(sigma) != 2:
            raise ValueError(f'sigma must be a tuple of length 2, but got length {len(sigma)}')
        if isinstance(dims, int):
            dims = (dims,)
        sigma = sorted(sigma)
        shape = list(signal.shape)
        flags = [True for _ in shape]
        for dim in dims:
            flags[dim] = False
        for i, flag in enumerate(flags):
            if flag:
                shape[i] = 1
        sigma = torch.rand(shape).to(signal) * (sigma[1] - sigma[0]) + sigma[0]
        noise = noise * sigma
    else:
        noise = sigma * noise
    return noise if noise_only else signal + noise


def poisson(signal: Ts, a: float | Ts, noise_only: bool = False) -> Ts:
    r"""
    Applying poissonian noise to signal:

    .. math::
        \tilde{\mathbf{x}}\sim\mathcal{P}(\mathbf{x}/a).

    .. attention::
        This is a non-differentiable operation.

    :param Tensor signal: Input signal.
    :param a: Scale factor :math:`a` in Poisson distribution.
        Must be broadcastable with ``signal`` if a Tensor.
    :type a: float | Tensor
    :param bool noise_only: If ``True``, return noise rather than noisy signal. Default: ``False``.
    :return: Noisy signal.
    :rtype: Tensor
    """
    noisy = torch.poisson(signal / a) * a
    return noisy - signal if noise_only else noisy
