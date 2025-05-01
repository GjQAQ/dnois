import torch

from ..base.typing import Numeric, Ts

__all__ = [
    'linear2srgb',
    'srgb2linear',
]


# reference: IEC 61966-2-1
def linear2srgb(image: Numeric, eps: float = 1e-8) -> float | Ts:
    """
    Convert signal ``image`` (typically an image tensor) from linear RGB space to sRGB space.

    :param image: Signal to be converted, assumed to be in range [0, 1].
    :type image: int or float or Tensor
    :param float eps: A small positive number to ensure numerical stability. Default: ``1e-8``.
    :return: Converted signal.
    :rtype: float or Tensor
    """
    a = 0.055
    image = image.clamp(eps, 1.)
    if torch.is_tensor(image):
        return torch.where(image <= 0.0031308, 12.92 * image, (1. + a) * image ** (1. / 2.4) - a)
    else:
        return 12.92 * image if image <= 0.0031308 else (1. + a) * image ** (1. / 2.4) - a


def srgb2linear(x: Numeric, eps: float = 1e-8) -> float | Ts:
    """
    Convert signal ``x`` (typically an image tensor) from sRGB space to linear RGB space.

    :param x: Signal to be converted, assumed to be in range [0, 1].
    :type x: int or float or Tensor
    :param float eps: A small positive number to ensure numerical stability. Default: ``1e-8``.
    :return: Converted signal.
    :rtype: float or Tensor
    """
    x = x.clamp(eps, 1.)
    if torch.is_tensor(x):
        return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    else:
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
