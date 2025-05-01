from typing import Literal

import torch
from torch import nn

from ..base.typing import Ts, Size2d, size2d

__all__ = [
    'cfa_collect',
    'cfa_flatten',
    'quantize',
    'rgb2raw',
    'spectral_integrate_cfa',

    'BayerPattern',
]

BayerPattern = Literal['RGGB', 'GRBG', 'BGGR', 'GBRG']


def cfa_flatten(image: Ts, unit_size: Size2d = 1) -> Ts:
    r"""
    Flatten the pixels in a channel-wise image with shape :math:`(\cdots,C,H,W)` into
    `CFA <https://en.wikipedia.org/wiki/Color_filter_array>`_ units with size :math:`(h,w)`
    to form an image with shape :math:`(\cdots,H\times h,W\times w)`, where :math:`C=hw`.
    This is the inverse of :py:func:`~dnois.sensor.cfa_collect`.

    .. note::

        This function is similar to :py:func:`~torch.nn.functional.pixel_shuffle` but
        removes a dimension of the input and supports unequal unit height and width.

    :param Tensor image: The image to be flattened, a tensor shape :math:`(\cdots,C,H,W)`.
    :param unit_size: Height and width of a pixel group :math:`(h,w)`. Default: ``(1, 1)``.
    :type: int or tuple[int, int]
    :return: Flattened image with shape :math:`(\cdots,H\times h,W\times w)`.
    :rtype: Tensor
    """
    us = size2d(unit_size)
    if image.size(-3) != us[0] * us[1]:
        raise ValueError(f'Number of channels must be equal to the product of unit height and width')
    if us[0] == us[1] == 1:
        return image.squeeze(-3)
    if us[0] == us[1]:
        return nn.functional.pixel_shuffle(image, us[0]).squeeze(-3)

    image = image.reshape(*image.shape[:-3], image.size(-3) // (us[0] * us[1]), *us, *image.shape[-2:])
    n = image.ndim
    image = image.permute(*list(range(n - 4)), n - 2, n - 4, n - 1, n - 3)
    image = image.reshape(*image.shape[:-4], image.size(-4) * us[0], image.size(-2) * us[1])
    return image.squeeze(-3)


def cfa_collect(image: Ts, unit_size: Size2d = 1) -> Ts:
    r"""
    Rearrange all the pixels in an image with shape :math:`(\cdots,H,W)` into channels-wise form
    :math:`(\cdots,C,H/h,W/w)`, where :math:`C=hw` is the number of channels and :math:`h` and
    :math:`w` are the size of a unit of a regular `color filter array (CFA)
    <https://en.wikipedia.org/wiki/Color_filter_array>`_. In this way, the vanilla 2D pixel
    array is divided into contiguous and non-overlapping pixel groups, or units.
    'Regular' means all the units have same size.

    The pixels in a unit will be rearranged into a single pixel with :math:`C=hw` channels
    in a row-major manner.

    .. note::

        This function is similar to :py:func:`~torch.nn.functional.pixel_unshuffle` but
        adds a new dimension to the input and supports unequal unit height and width.

    :param Tensor image: The image to be rearranged, a tensor shape :math:`(\cdots,H,W)`.
    :param unit_size: Height and width of a pixel group :math:`(h,w)`. Default: ``(1, 1)``.
    :type: int or tuple[int, int]
    :return: Rearranged image with shape :math:`(\cdots,hw,H/h,W/w)`.
    :rtype: Tensor
    """
    us = size2d(unit_size)
    image = image.unsqueeze(-3)  # ... x 1 x H x W
    if us[0] == us[1] == 1:
        return image
    if us[0] == us[1]:
        return nn.functional.pixel_unshuffle(image, us[0])

    image = image.reshape(
        *image.shape[:-2], image.size(-2) // us[0], us[0], image.size(-1) // us[1], us[1]
    )
    n = image.ndim
    image = image.permute(*list(range(n - 4)), n - 3, n - 1, n - 4, n - 2)
    image = image.reshape(*image.shape[:-5], image.size(-5) * us[0] * us[1], *image.shape[-2:])
    return image


# kornia.color
def rgb2raw(image: Ts, pattern: BayerPattern) -> Ts:
    """
    Convert an RGB image into a single-channel image using Bayer CFA pattern.

    :param Tensor image: The RGB image, a tensor of shape ``(..., 3, H, W)``.
    :param BayerPattern pattern: Bayer CFA pattern, either ``'RGGB'``, ``'GRBG'``,
        ``'BGGR'`` or ``'GBRG'``, specifying how are pixels arranged in the order
        of upper left, upper right, lower left, lower right.
    :return: A single-channel image with shape ``(..., 1, H, W)``.
    :rtype: Tensor
    """
    output: Ts = image[..., 1:2, :, :].clone()

    if pattern == 'RGGB':
        output[..., :, ::2, ::2] = image[..., 0:1, ::2, ::2]  # red
        output[..., :, 1::2, 1::2] = image[..., 2:3, 1::2, 1::2]  # blue
    elif pattern == 'GRBG':
        output[..., :, ::2, 1::2] = image[..., 0:1, ::2, 1::2]  # red
        output[..., :, 1::2, ::2] = image[..., 2:3, 1::2, ::2]  # blue
    elif pattern == 'BGGR':
        output[..., :, 1::2, 1::2] = image[..., 0:1, 1::2, 1::2]  # red
        output[..., :, ::2, ::2] = image[..., 2:3, ::2, ::2]  # blue
    elif pattern == 'GBRG':
        output[..., :, 1::2, ::2] = image[..., 0:1, 1::2, ::2]  # red
        output[..., :, ::2, 1::2] = image[..., 2:3, ::2, 1::2]  # blue

    return output


def spectral_integrate_cfa(
    radiance: Ts,
    srf: Ts,
    unit_size: Size2d = 1,
    channel_dim: bool = False
) -> Ts:
    r"""
    Integrate given radiance field across wavelengths with given spectral response
    function (SRF).

    This function supports regular `color filter array (CFA)
    <https://en.wikipedia.org/wiki/Color_filter_array>`_, where the vanilla 2D pixel
    array is divided into contiguous and non-overlapping pixel groups. 'Regular' means
    all the pixel groups are identical.
    See :py:func:`cfa_collect` and :py:func:`cfa_flatten` for more details.

    :param Tensor radiance: A tensor of shape :math:`(\cdots,N_\lambda,H,W)`.
    :param Tensor srf: A tensor of shape :math:`(N_C, N_\lambda)`.
    :param unit_size: Height and width of a pixel group, in pixels. Note that
        their product should be equal to :math:`N_C`.
        Default: ``(1, 1)``.
    :type: int or tuple[int, int]
    :param bool channel_dim: If ``True``, last three dimension of returned tensor will be
        ``(N_C, H // unit_size[0], W // unit_size[1])``; otherwise ``(H, W)``.
        Default: ``False``.
    :return: Integrated radiance field, of shape
        ``(..., N_C, H // unit_size[0], W // unit_size[1])`` or ``(..., H, W)``.
    :rtype: Tensor
    """
    unit_size = size2d(unit_size)
    if radiance.size(-3) != srf.size(-1):
        raise ValueError(f'N_wl of radiance is {radiance.size(-3)} but that of srf is {srf.size(-1)}')
    if unit_size[0] * unit_size[1] != srf.size(-2):
        raise ValueError(f'Pixel group size is {unit_size} but number of channels is {srf.size(-2)}')
    if radiance.size(-2) % unit_size[0] != 0 or radiance.size(-1) % unit_size[1] != 0:
        raise ValueError(f'Spatial size of radiance ({radiance.shape[-2:]}) '
                         f'must be divisible by pixel group size ({unit_size})')

    unit_size = size2d(unit_size)
    radiance = cfa_collect(radiance, unit_size)  # ... x N_wl x N_C x H' x W'
    radiance = radiance.transpose(-4, -3)  # ... x N_C x N_wl x H' x W'
    t = torch.einsum('...wij,...w->...ij', radiance, srf)  # ... x N_C x H' x W'

    if channel_dim:
        return t
    else:
        return cfa_flatten(t, unit_size).squeeze(-3)  # ... x H x W


def quantize(signal: Ts, levels: int = 256, differentiable: bool = False) -> Ts:
    """
    Quantize continuous-valued signal, emulating an analogous-to-digital conversion.

    :param Tensor signal: The signal to be quantized whose value must be
        in :math:`[0,1]`.
    :param int levels: Quantization levels. 256 for example, which is the number
        of levels for most image sensors. Default: 256.
    :param bool differentiable: Whether to perform quantization
        in a differentiable manner. Specifically, if ``True``,
        a quantization noise will be added to signal to simulate quantization.
        Default: ``False``.
    :return: Quantized signal.
    :rtype: Tensor
    """
    if signal.min().item() < 0 or signal.max().item() > 1:
        raise ValueError(f'Value of signal must be in [0, 1]')
    v_max = levels - 1
    qt = signal * v_max
    qt = torch.round(qt) / v_max
    if differentiable:
        qt_noise = qt - signal.detach()
        return signal + qt_noise
    else:
        return qt
