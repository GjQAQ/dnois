import torch

from .. import torch as _t
from ..base.typing import Ts

__all__ = [
    'wiener',
]


def wiener(signal: Ts, kernel: Ts, nsr: float | Ts, ndim: int = 2) -> Ts:
    r"""
    Wiener deconvolution:

    .. math::
        \hat{x} = \mathcal{F}^{-1}\left\{\frac{h^*}{|h|^2 + \sigma^2}\mathcal{F}\{y\}\right\}

    :param Tensor signal: Signal to be deconvolved :math:`y`.
    :param Tensor kernel: Kernel used for deconvolution.
    :param nsr: Noise-to-signal ratio :math:`\sigma`.
    :type nsr: float or Tensor
    :param int ndim: Number of dimensions of the signal.
    :return: Deconvolved signal.
    """
    dims = list(range(-ndim, 0))
    signal_ft = torch.fft.fftn(torch.fft.ifftshift(signal, dim=dims), dim=dims)
    kernel_ft = torch.fft.fftn(torch.fft.ifftshift(kernel, dim=dims), s=signal.shape[-ndim:])
    filtered_ft = signal_ft * kernel_ft.conj() / (_t.abs2(kernel_ft) + nsr ** 2)
    filtered = torch.fft.ifftn(filtered_ft, dim=dims).real
    filtered = torch.fft.fftshift(filtered, dim=dims)
    return filtered
