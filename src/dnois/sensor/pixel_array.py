import torch

from .. import utils
from ..base.typing import Pair, Size2d, Ts, size2d, pair

__all__ = [
    'Sensor',
]


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
