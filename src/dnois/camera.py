import torch

from .base.typing import Ts, Callable
from .optics import Optics
from .scene import Scene
from .sensor import Sensor

__all__ = [
    'Camera',
]


class Camera(torch.nn.Module):
    __call__: Callable[..., Ts]

    def __init__(self, optics: Optics, sensor: Sensor):
        super().__init__()
        self.optics = optics
        self.sensor = sensor

    def forward(self, scene: Scene, optics_kw: dict = None, sensor_kw: dict = None) -> Ts:
        imaged_radiance_field = self.optics(scene, **(optics_kw or {}))
        captured = self.sensor(imaged_radiance_field, **(sensor_kw or {}))
        return captured
