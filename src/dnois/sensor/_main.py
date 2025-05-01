import torch

from ._func import *
from .noise import gaussian
from .. import utils, isp
from ..base.typing import Pair, Size2d, Ts, size2d, pair

__all__ = [
    'Sensor',
    'StandardSensor',
]

_MSG1 = 'Number of channels of input radiance is not {0} for an {1} sensor without SRF'


def _make_srf(srf: Ts | tuple[Ts, Ts, Ts], pattern: BayerPattern) -> Ts | None:
    if srf is None:
        return None
    r, g, b = srf
    if pattern == 'RGGB':
        srf = (r, g, g, b)
    elif pattern == 'GRBG':
        srf = (g, r, b, g)
    elif pattern == 'BGGR':
        srf = (b, g, g, r)
    elif pattern == 'GBRG':
        srf = (g, b, r, g)
    else:
        raise ValueError(f'Wrong bayer pattern: {pattern}')
    return torch.stack(srf)


class Sensor(torch.nn.Module):
    """
    A basic sensor model.

    :param pixel_num: Numbers of pixels in vertical and horizontal directions.
    :type pixel_num: int or tuple[int, int]
    :param pixel_size: Height and width of a pixel in meters.
    :type pixel_size: float or tuple[float, float]
    """

    def __init__(self, pixel_num: Size2d, pixel_size: Pair[float]):
        pixel_num = size2d(pixel_num)
        pixel_size = pair(pixel_size, float)
        utils.check.positive(pixel_num, 'pixel_num')
        utils.check.positive(pixel_size, 'pixel_size')

        super().__init__()
        #: Numbers of pixels in vertical and horizontal directions.
        self.pixel_num: tuple[int, int] = pixel_num
        #: Height and width of a pixel in meters.
        self.pixel_size: tuple[float, float] = pixel_size

    def forward(self, radiance: Ts) -> Ts:
        raise NotImplementedError(f'{type(self).__name__} cannot be used for imaging')

    @property
    def size(self) -> tuple[float, float]:
        """
        Returns the physical size i.e. height and width of the sensor.

        :type: tuple[float, float]
        """
        return self.pixel_size[0] * self.pixel_num[0], self.pixel_size[1] * self.pixel_num[1]

    @property
    def h(self):
        """Physical height of the sensor in meters.\n\n:type: float"""
        return self.pixel_size[0] * self.pixel_num[0]

    @property
    def w(self):
        """Physical width of the sensor in meters.\n\n:type: float"""
        return self.pixel_size[1] * self.pixel_num[1]


class StandardSensor(Sensor):
    """
    A simple RGB or grayscale sensor model, which processes the radiance field reaching the sensor
    plane as follows:

    #.  Divide pixels into channels according to Bayer CFA.
    #.  Spectral integral by ``srf``.
    #.  Apply additional Gaussian white noise.
    #.  Restrict signal values to a given range.
    #.  Quantize signal to a given level.

    :param pixel_num: Numbers of pixels in vertical and horizontal directions.
    :type pixel_num: int or tuple[int, int]
    :param pixel_size: Height and width of a pixel in meters.
    :type pixel_size: float or tuple[float, float]
    :param bool rgb: RGB sensor if ``True``, grayscale sensor otherwise. Default: ``True``.
    :param srf: SRF tensor of shape ``(N_C, N_wl)`` where ``N_C`` is 1 or 3,
        or a 3-tuple of tensors of length ``N_wl``, corresponding to the SRF of
        R, G, B channels. Default: do not perform spectral integral.
    :type srf: Tensor or tuple[Tensor, Tensor, Tensor]
    :param BayerPattern bayer_pattern: See :py:func:`rgb2raw`. Default: no CFA.
    :param noise_std: Standard deviation of the Gaussian white noise.
        See :func:`gaussian`. Default: 0.
    :type noise_std: float or tuple[float, float]
    :param float max_value: Maximum possible value of output signal. Default: 1.
    :param int _quantize: Quantization level. The output signal will not be
        quantized if a negative or zero value is given. Default: 256.
    :param bool linear2srgb: Whether to convert linear RGB to sRGB. Default: ``True``.
    :param bool differentiable_quant: See :py:func:`quantize`. Default: ``True``.
    """

    #: Spectral response functions. Either ``None`` or a tensor of shape
    #: ``(1, N_wl)`` or ``(4, N_wl)`` (two identical green SRF in Bayer CFA)
    srf: Ts

    def __init__(
        self,
        pixel_num: Size2d,
        pixel_size: Pair[float],
        rgb: bool = True,
        srf: Ts | tuple[Ts, Ts, Ts] = None,
        bayer_pattern: BayerPattern = None,
        noise_std: float | tuple[float, float] = 0.,
        max_value: float = 1.,
        _quantize: int = 256,
        linear2srgb: bool = True,
        differentiable_quant: bool = True,
    ):
        super().__init__(pixel_num, pixel_size)
        self.rgb: bool = rgb  #: RGB sensor or not.
        #: Bayer CFA pattern. See :py:func:`rgb2raw`
        self.bayer_pattern: BayerPattern | None = bayer_pattern
        self.noise_std: float | tuple[float, float] = noise_std  #: Standard deviation of Gaussian noise.
        self.max_value: float = max_value  #: Maximum possible value of signal.
        self.quantize: int = _quantize  #: Quantization level.
        self.linear2srgb: bool = linear2srgb  #: Whether to convert linear RGB to sRGB.
        #: Whether to perform differentiable quantization.
        self.differentiable_quantization: bool = differentiable_quant

        if not (
            srf is None or
            (isinstance(srf, tuple) and len(srf) == 3) or
            (torch.is_tensor(srf) and srf.size(-2) not in (1, 3))
        ):
            raise ValueError(
                f'Number of channels of SRF must be one or three in {self.__class__.__name__}'
            )
        if rgb and bayer_pattern is not None:
            srf = _make_srf(srf, bayer_pattern)  # 4 x N_wl
        self.register_buffer('srf', srf)

    def forward(self, radiance: Ts) -> Ts:
        """
        Simulate the process of conversion from optical signal ``radiance`` to
        electric signal, an 2D image.

        :param Tensor radiance: A tensor representing the received radiance field,
            of shape ``(..., N_wl, H, W)``.
        :return: Output image signal, a tensor of shape ``(..., N_C', H, W)``.
            If ``self.srf`` is not ``None``, ``radiance`` is monochromatic (``N_wl=1``)
            or ``self.bayer_pattern`` is not ``None``, ``N_C'`` is 1. Otherwise,
            ``N_C'`` is 3 (RGB regardless of SRF and Bayer CFA).
        :rtype: Tensor
        """
        if self.srf is None:
            if self.rgb:  # radiance: ... x 3 x H x W
                if radiance.size(-3) != 3:
                    raise ValueError(_MSG1.format('3', 'RGB'))
                if self.bayer_pattern is None:
                    transmitted = radiance  # ... x 3 x H x W
                else:
                    transmitted = rgb2raw(radiance, self.bayer_pattern).squeeze(-3)  # ... x 1 x H x W
            else:  # radiance:... x 1 x H x W
                if radiance.size(-3) != 1:
                    raise ValueError(_MSG1.format('1', 'grayscale'))
                transmitted = radiance  # ... x 1 x H x W
        else:
            if radiance.size(-3) != self.srf.size(-1):
                raise ValueError(
                    'Number of wavelengths of input radiance is not equal to '
                    'that of the sensor\'s SRF'
                )
            unit_size = 2 if self.rgb else 1
            transmitted = spectral_integrate_cfa(radiance, self.srf, unit_size)  # ... x H x W
            transmitted = transmitted.unsqueeze(-3)  # ... x 1 x H x W

        signal = gaussian(transmitted, self.noise_std)
        signal = signal.clip(0., self.max_value)

        if self.linear2srgb:
            signal = isp.linear2srgb(signal / self.max_value) * self.max_value

        if self.quantize <= 0:
            return signal
        return quantize(
            signal.detach() / self.max_value, self.quantize, self.differentiable_quantization
        )
