import abc

from . import surf, ray
from .. import system, formation
from ... import utils, scene as _sc
from ...base.typing import Ts, Vector

__all__ = [
    'ForwardRayTracingOptics',
]


# Currently this class is used exclusively by dnois.rt.CoaxialRayTracing
# so its code couples with that class closely
class ForwardRayTracingOptics(system.ImagingOptics, system.RenderImageSceneMixIn, metaclass=abc.ABCMeta):
    surfaces: surf.SurfaceList

    @abc.abstractmethod
    def trace_point(self, point: Ts, wl: Vector = None, sampler: surf.Sampler = None) -> ray.BatchedRay:
        """
        Trace a group of rays emitted from ``point`` through surfaces until the image plane.

        :param Tensor point: Coordinate of points in
            :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`.
            A tensor of shape ``(..., 3)``.
        :param wl: Wavelengths of rays. A float, a sequence of float or a tensor of shape ``(N_wl,)``.
        :param Callable sampler: A callable object whose signature is described by
            :meth:`dnois.optics.rt.Aperture.sampler`. This is typically created by this method as well.
        :return: Rays after tracing with shape ``(.., N_wl, N_spp)``. Their origins are located at the image plane.
        :rtype: ray.BatchedRay
        """
        pass

    @utils.with_external
    def render_image_scene(
        self,
        scene: _sc.ImageScene,
        wl: Ts = None,
        depth: Ts | tuple[Ts, Ts] = None,
        sampler: surf.Sampler = None,
        vignette: bool = True,
        repetitions: int = 1,
    ) -> Ts:
        rendered = self._render_image_scene(scene, wl, depth, sampler, vignette)
        if repetitions > 1:
            for _ in range(repetitions - 1):
                rendered += self._render_image_scene(scene, wl, depth, sampler, vignette)
            rendered /= repetitions
        return rendered

    @utils.with_external
    def render_point_cloud_scene(
        self,
        scene: _sc.PointCloudScene,
        wl: Ts = None,
        sampler: surf.Sampler = None,
        vignette: bool = True,
        repetitions: int = 1,
    ) -> Ts:
        rendered = self._render_point_cloud_scene(scene, wl, sampler, vignette)
        if repetitions > 1:
            for _ in range(repetitions - 1):
                rendered += self._render_point_cloud_scene(scene, wl, sampler, vignette)
            rendered /= repetitions
        return rendered

    def _render_image_scene(
        self, scene: _sc.ImageScene, wl: Ts, depth: Ts | tuple[Ts, Ts], sampler: surf.Sampler, vignette: bool
    ) -> Ts:
        if wl.numel() != scene.n_wl:
            raise ValueError(f'A scene with {self.wl.numel()} wavelengths expected, got {scene.n_wl}')
        if len(self.surfaces) == 0:
            raise RuntimeError(f'No surface available')

        scene = scene.batch()
        n_b, n_wl, n_h, n_w = scene.image.shape

        depth_map = self._make_depth_map(scene, depth)  # (B, H, W)
        o = self.points_grid((n_h, n_w), depth_map, True)  # (B, H, W, 3)
        o = o.flatten(1, 2)  # (B, H*W, 3)

        out_ray = self.trace_point(o, wl, sampler)  # (B, H*W, N_wl, spp)
        xy = out_ray.o[..., :2].transpose(1, 2)  # (B, N_wl, H*W, spp, 2)
        valid = out_ray.valid.transpose(1, 2)  # (B, N_wl, H*W, spp)
        value = scene.image.flatten(-2, -1)  # (B, N_wl, H*W)
        return self._spots2image(value, xy, valid, vignette)  # (B, N_wl, H, W)

    def _render_point_cloud_scene(
        self, scene: _sc.PointCloudScene, wl: Ts, sampler: surf.Sampler, vignette: bool
    ) -> Ts:
        if wl.numel() != scene.n_wl:
            raise ValueError(f'A scene with {wl.numel()} wavelengths expected, got {scene.n_wl}')
        if len(self.surfaces) == 0:
            raise RuntimeError(f'No surface present in optics')

        o = scene.locations  # (N, 3)
        out_ray = self.trace_point(o, wl, sampler)  # (N, N_wl, spp)
        xy = out_ray.o[..., :2].transpose(0, 1)  # (N_wl, N, spp, 2)
        valid = out_ray.valid.transpose(0, 1)  # (N_wl, N, spp)
        value = scene.luminance  # (N_wl, N)
        image = self._spots2image(value, xy, valid, vignette)  # (N_wl, H, W)
        return image

    def _spots2image(self, value, xy, valid, vignette):
        spp = valid.shape[-1]
        n_h, n_w = self.sensor.pixel_num
        x, y = xy[..., 0] / -self.sensor.pixel_size[1], xy[..., 1] / self.sensor.pixel_size[0]
        x, y = x + n_h / 2, y + n_w / 2  # (B, N_wl, H*W, spp)
        if not vignette:
            value = value * spp / (valid.sum(-1) + 1e-5)
        image = formation.spots2image((n_h, n_w), y, x, value, valid)  # (B, N_wl, H, W)
        return image
