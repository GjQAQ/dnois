from .. import base
from ..base.typing import Literal, Ts, cast

__all__ = [
    'fdc2rgb',
    't4plot',
    'wl2rgb',

    'RGBFormat',
    'RGBTriplet',
]

RGBTriplet = tuple[float, float, float] | str
RGBFormat = Literal['floats', 'ints', 'hex']


def t4plot(tensor: Ts) -> Ts:
    return tensor.detach().cpu()


def wl2rgb(wl: float, gamma: float = 0.8, output_format: RGBFormat = 'floats') -> RGBTriplet:
    """
    Convert wavelength to RGB color. The output color will be limited to
    purple if the wavelength is less than 380 nm and to red if the wavelength
    is greater than 780 nm.

    :param float wl: Wavelength value.
    :param float gamma: Gamma value.
    :param str output_format: Output format. Choices:

        ``'floats'``
            Return a 3-tuple of floats in the range [0, 1].

        ``'ints'``
            Return a 3-tuple of integers in the range [0, 255].

        ``'hex'``
            Return a hex string in the format ``'#rrggbb'``. It can be used
            in matplotlib without extra processing.
    :return: RGB color. See ``output_format`` for details.
    """
    wl = base.Length.default_to(wl, 'nm')
    if wl < 380:
        red, green, blue = 1., 0., 1.
    elif 380 <= wl <= 440:
        red, green, blue = -(wl - 440) / (440 - 380), 0., 1.
    elif 440 <= wl <= 490:
        red, green, blue = 0.0, (wl - 440) / (490 - 440), 1.
    elif 490 <= wl <= 510:
        red, green, blue = 0.0, 1., -(wl - 510) / (510 - 490)
    elif 510 <= wl <= 580:
        red, green, blue = (wl - 510) / (580 - 510), 1., 0.
    elif 580 <= wl <= 645:
        red, green, blue = 1.0, -(wl - 645) / (645 - 580), 0.
    else:
        red, green, blue = 1., 0., 0.

    if wl < 380:
        factor = 0.3
    elif 380 <= wl <= 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif 420 <= wl <= 700:
        factor = 1.0
    elif 700 <= wl <= 780:
        factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)
    else:
        factor = 0.3

    rgb = ((red * factor) ** gamma, (green * factor) ** gamma, (blue * factor) ** gamma)
    if output_format == 'floats':
        return rgb
    else:
        rgb = tuple(int(v * 255) for v in rgb)
        if output_format == 'ints':
            return cast(RGBTriplet, rgb)
        elif output_format == 'hex':
            return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
        else:
            raise ValueError(f'Unknown output format: {output_format}')


def fdc2rgb(image: Ts, **kwargs) -> Ts:
    """
    Convert an image tensor of shape ``(..., 3, H, W)`` where the ``3`` dimension
    means some "images" corresponding to Fraunhofer F, d and C lines to RGB color.
    In other words, shape of output tensor is ``(..., 3, H, W)`` still but the
    ``3`` dimension means RGB color.

    :param Tensor image: Image tensor.
    :param kwargs: Additional keyword arguments passed to :func:`wl2rgb`.
    :return: RGB image tensor.
    :rtype: Tensor
    """
    wls = base.fdc()
    color = [image.new_tensor(wl2rgb(wl, **kwargs)) for wl in wls]

    imgs = image.unbind(-3)
    imgs = [img.unsqueeze(-3) * clr.reshape(3, 1, 1) for img, clr in zip(imgs, color)]
    image = cast(Ts, sum(imgs))

    mv = image.max().item()
    if mv > 1.:
        image = image / mv
    return image
