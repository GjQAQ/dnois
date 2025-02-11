import torch

from .base.typing import Ts, Callable
from .optics import ImagingOptics
from .scene import Scene
from .sensor import Sensor

__all__ = [
    'Camera',
]


class Camera(torch.nn.Module):
    """
    A basic camera model.

    :param ImagingOptics optics: The imaging optics.
    :param Sensor sensor: The sensor.
    """
    __call__: Callable[..., Ts]

    def __init__(self, optics: ImagingOptics, sensor: Sensor):
        super().__init__()
        self.optics: ImagingOptics = optics  #: The imaging optics.
        self.sensor: Sensor = sensor  #: The sensor.

    def forward(self, scene: Scene, optics_kw: dict = None, sensor_kw: dict = None) -> Ts:
        """
        Simulate the capture a scene.

        :param Scene scene:
        :param dict optics_kw: Keyword arguments passed to :attr:`.optics`.
        :param dict sensor_kw: Keyword arguments passed to :attr:`.sensor`.
        :return: Simulated captured image.
        :rtype: Ts
        """
        imaged_radiance_field = self.optics(scene, **(optics_kw or {}))
        captured = self.sensor(imaged_radiance_field, **(sensor_kw or {}))
        return captured
