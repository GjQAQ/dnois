from .core import *

__all__ = [
    'AverageFov',
    'ChiefRayFov',
    'FixedFov',
    'PerspectiveFov',
]


class FixedFov(CrtFovModel):
    type = 'fixed'

    def __init__(self, fov: float | tuple[float, float] | tuple[float, float, float, float]):
        if isinstance(fov, int):
            fov = float(fov)
        if isinstance(fov, float):
            fov = (fov, fov)  # in x and y direction
        if len(fov) == 2:
            fov = (-fov[0], fov[0], -fov[1], fov[1])  # (x1, x2, y1, y2)
        self.fov = {
            'x_lower': fov[0],
            'x_upper': fov[1],
            'y_lower': fov[2],
            'y_upper': fov[3]
        }

    def get(self, optics: 'CoaxialRayTracing', which: FovItem) -> float:
        return self.fov[which]


class PerspectiveFov(CrtFovModel):
    type = 'perspective'

    def get(self, optics: 'CoaxialRayTracing', which: FovItem) -> float:
        return getattr(optics.reference, f'fov_{which}')


class ChiefRayFov(CrtFovModel):
    type = 'chief'

    def get(self, optics: 'CoaxialRayTracing', which: FovItem) -> float:
        raise NotImplementedError()


class AverageFov(CrtFovModel):
    type = 'average'

    def get(self, optics: 'CoaxialRayTracing', which: FovItem) -> float:
        raise NotImplementedError()
