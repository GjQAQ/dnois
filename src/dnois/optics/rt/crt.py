import math
import warnings

import torch

from . import surf, SurfaceList
from .ray import BatchedRay
from .. import system, _func
from ... import scene as _sc, base, utils, fourier, torch as _t, ext
from ...base import typing, ddb
from ...base.typing import Ts, Any, Size2d, Vector, Scalar, Self
from ...sensor import Sensor

__all__ = [
    'CoaxialRayTracing',
]

DEFAULT_FIND_CHIEF_SAMPLES: int = 101
DEFAULT_SAMPLES: int = 512
FOV_THRESHOLD4CHIEF_RAY = math.radians(0.01)

PsfCenter = typing.Literal['linear', 'mean', 'chief']
PsfType = typing.Literal['inc_rect', 'inc_gaussian', 'coh_kirchoff', 'coh_huygens', 'coh_fraunhofer']
FovType = typing.Literal['perspective', 'chief', 'average']
PupilType = typing.Literal['probe', 'trace', 'paraxial']
ChiefSide = typing.Literal['obj', 'img', 'object', 'image']
WlReduction = typing.Literal['none', 'mean', 'center']
PupilSpec = typing.Double[Ts]  # radius, z-coordinate


def _check_arg(arg: Any, n1: str, n2: str):
    if arg is not None:
        warnings.warn(f'{n1} and {n2} are given simultaneously, {n2} will be ignored')


def _plot_set_ax(ax, x_range: float):
    ax.set_xlim(-x_range * 0.05, x_range * 1.05)
    ax.set_xlabel('$z/m$')
    # ax.set_xticks([])
    ax.set_ylabel('$y/m$')
    # ax.set_yticks([])
    ax.set_position([0.1, 0.1, 0.9, 0.9])
    ax.set_aspect('equal')


def _plot_linestyles(n: int) -> list[str]:
    bases = ['-', '--', '-.', ':']
    segs, rem = divmod(n, len(bases))
    if segs > 0:
        warnings.warn('More than 4 linestyles are being used. Some linestyles may be repeated')
    lss = bases * segs + bases[:rem]
    return lss


def _plot_rays_3d(ax, start: Ts, end: Ts, valid: Ts, colors: list[str], lss: list[str]):
    # shape: N_fov x N_wl x N_spp x 3
    for ls, fov_slc1, fov_slc2, v in zip(lss, start, end, valid):  # N_wl x N_spp x 3
        for clr, wl_slc1, wl_slc2, vv in zip(colors, fov_slc1, fov_slc2, v):  # N_spp x 3
            ax.plot(
                (utils.t4plot(wl_slc1[:, 2][vv]), utils.t4plot(wl_slc2[:, 2][vv])),
                (utils.t4plot(wl_slc1[:, 1][vv]), utils.t4plot(wl_slc2[:, 1][vv])),
                color=clr, linestyle=ls, linewidth=0.75
            )


def _make_direction(sampled_point: Ts, origin: Ts, normalize: bool = False) -> tuple[Ts, Ts | None]:
    # Typically, to create rays, some origins in object space are selected
    # and some points are sampled on the first surface in a system. The directions
    # of rays are thus the vectors pointing to sampled points from origins.
    # However, the origins may be located at infinity. In that case, the x and
    # y coordinates of origins are assumed to be finite and serve as the tangents
    # of x and y FoV, respectively.
    is_inf = origin[..., 2].isinf()
    if is_inf.all():
        d = torch.cat([-origin[..., :2], torch.ones_like(origin[..., [2]])], -1)
    else:
        d = sampled_point - origin
        if is_inf.any():
            d = torch.where(
                is_inf.unsqueeze(-1),
                torch.cat([-origin[..., :2], torch.ones_like(origin[..., [2]])], -1),
                d
            )
    if normalize:
        length = d.norm(2, -1)
        return d / length.unsqueeze(-1), length
    else:
        return d, None


if ext.vis.mpl_available():
    import matplotlib.pyplot as plt


    class CoaxialRayTracingVisMixIn:
        @torch.no_grad()
        @utils.with_external
        def plot_spot_diagram(
            self: 'CoaxialRayTracing', points: Ts = None, wl: Vector = None, *, width=None
        ) -> plt.Figure:
            self._check_circ_aperture()
            self._check_circ_surf()

            if points is None:
                fov_half = self.reference.fov_half
                fov = [0., fov_half * 0.5 ** 0.5, fov_half]
                points = self.fovd2obj([(0., fov_item) for fov_item in fov], float('inf'))
            wl = self.pick('wl', wl)
            wl = typing.vector(wl, device=self.device, dtype=self.dtype)

            points = self.cam2lens(points)
            n_point = points.size(0)
            n_row = int(math.sqrt(n_point) + 1e-5)
            n_col = int(math.ceil(n_point / n_row))
            fig, axs = plt.subplots(n_row, n_col, squeeze=False, figsize=(n_col * 5, n_row * 5))

            entr_r, entr_z = self.entr_pupil('paraxial', wl, 'center')
            entr_d, entr_z = entr_r.item() * 2, entr_z.item()

            pupil_ap = surf.CircularAperture(entr_d)
            pupil_ap.to(device=self.device, dtype=self.dtype)
            x, y = pupil_ap.sample_unipolar(6, 6)
            pupil_points = torch.stack([x, y, torch.full_like(x, entr_z)], -1)  # N_spp x 3
            entr_center = self.new_tensor([0, 0, entr_z])

            for i in range(n_point):
                # _, chief_ray = self._generate_rays(points[i], wl, 1)
                # radial_offset = torch.sqrt(chief_ray.o[..., :2].square().sum(-1))
                # d_proj = torch.sqrt(chief_ray.d[..., :2].square().sum(-1))
                # rs_roc = radial_offset / d_proj  # N_wl x 1
                # enter_pupil_z = torch.sqrt(rs_roc.square() - radial_offset.square()).squeeze(-1)  # N_wl
                # enter_pupil_z += chief_ray.z
                direction, _ = _make_direction(pupil_points, points[i])  # N_spp|1 x 3
                ray_in = BatchedRay(pupil_points, direction, wl.view(-1, 1))  # N_wl x N_spp
                ray_out = self.trace_ray(ray_in)  # N_wl x N_spp

                chief_direction, _ = _make_direction(entr_center, points[i])  # 3
                chief_ray_in = BatchedRay(entr_center, chief_direction, wl)  # N_wl
                chief_ray_out = self.trace_ray(chief_ray_in)  # N_wl

                r, c = i // n_col, i % n_col
                axs: list[list[plt.Axes]]
                ax: plt.Axes = axs[r][c]
                for j in range(wl.size(0)):
                    x, y = utils.t4plot(ray_out.x[j] - chief_ray_out.x[j]), utils.t4plot(
                        ray_out.y[j] - chief_ray_out.y[j])
                    ax.scatter(
                        x, y,
                        s=2,
                        c=utils.wl2rgb(wl[j].item(), output_format='hex'),
                        label=f'{wl[j].item():.4g}',
                    )
                    ax.legend()
                    ax.set_aspect('equal')
                    ax.set_xlim(-width, width)
                    ax.set_ylim(-width, width)

            return fig

        @torch.no_grad()
        @utils.with_external
        def plot_cross_section(
            self: 'CoaxialRayTracing',
            fig: plt.Figure = None,
            fov: Vector = None,
            wl: Vector = None,
            init_rays: int = 10,
        ) -> tuple[plt.Figure, plt.Axes]:
            self._check_circ_aperture()
            self._check_circ_surf()

            if fig is None:
                fig, ax = plt.subplots(figsize=(12.8, 9.6), subplot_kw={'frameon': True})
            else:
                ax = fig.axes[0][0]
            if fov is None:
                fov_half = self.reference.fov_half
                fov = [0., fov_half * 0.5 ** 0.5, fov_half]
            fov = typing.vector(fov, device=self.device, dtype=self.dtype)

            self._plot_components(ax)

            # image_plane
            if self.sensor is not None:
                diag_length = (self.sensor.h ** 2 + self.sensor.w ** 2) ** 0.5
                y = torch.linspace(-diag_length / 2, diag_length / 2, 100, device=self.device)
                ax.plot(
                    utils.t4plot(torch.full_like(y, self.surfaces.total_length.item())), utils.t4plot(y),
                    color='black', linewidth=2
                )

            # rays
            o = self.surfaces.first.sample('diameter', init_rays, torch.pi / 2)  # N_spp x 3
            d = torch.stack([torch.zeros_like(fov), fov.tan(), torch.ones_like(fov)], dim=-1)
            d = d.unsqueeze(1).unsqueeze(1)  # N_fov x 1 x 1 x 3
            ray = BatchedRay(o, d, wl.reshape(1, -1, 1))  # N_fov x N_wl x N_spp
            self._plot_rays(ax, ray, fov, wl)

            _plot_set_ax(ax, self.surfaces.total_length.item())
            return fig, ax

        @torch.no_grad()
        def plot_psf_map(
            self: 'CoaxialRayTracing',
            depth: float = float('inf'),
        ) -> plt.Figure:
            pass

        def _plot_components(self: 'CoaxialRayTracing', ax, points: int = 100):
            edge_z = []
            for sf in self.surfaces:  # surfaces
                sf: surf.CircularSurface
                if isinstance(sf, surf.CircularStop):
                    r = sf.apt.radius.item()
                    length = r / 5
                    z = sf.ctx.baseline.item()
                    ax.plot(
                        [[z, z, z - length / 2, z - length / 2], [z, z, z + length / 2, z + length / 2]],
                        [[r, -r, r, -r], [r + length, -r - length, r, -r]],
                        color='black', linewidth=1
                    )
                    edge_z.append(None)
                else:
                    y = torch.linspace(-sf.apt.radius, sf.apt.radius, points, device=self.device)
                    z = sf.h_extended(torch.zeros_like(y), y) + sf.ctx.baseline
                    ax.plot(utils.t4plot(z), utils.t4plot(y), color='black', linewidth=1)
                    edge_z.append(z[-1].item())
            for i in range(len(self.surfaces) - 1):  # edges
                if self.surfaces[i].material.name == 'vacuum':
                    continue

                r1 = self.surfaces[i].apt.radius.item()
                r2 = self.surfaces[i + 1].apt.radius.item()
                r = max(r1, r2)
                ax.plot(
                    [[edge_z[i], edge_z[i]], [edge_z[i + 1], edge_z[i + 1]]],
                    [[r, -r], [r, -r]],
                    color='black', linewidth=1
                )
                if r1 != r2:
                    if r1 > r2:
                        z = edge_z[i + 1]
                    else:
                        z = edge_z[i]
                        r1, r2 = r2, r1
                    ax.plot([[z, z], [z, z]], [[r2, -r2], [r1, -r1]], color='black', linewidth=1)

        def _plot_rays(self: 'CoaxialRayTracing', ax, ray: BatchedRay, fov: Ts, wl: Ts):  # ray: N_fov x N_wl x N_spp
            ray.broadcast_().march_to_(ray.new_tensor(0.))
            colors = [utils.wl2rgb(_wl, output_format='hex') for _wl in wl.tolist()]
            lss = _plot_linestyles(fov.numel())

            rays_record = []
            for sf in self.surfaces:
                out_ray = sf(ray)
                rays_record.append(out_ray.broadcast_())
                ray = out_ray
            out_ray = ray.march_to(self.surfaces.total_length)
            rays_record.append(out_ray.broadcast_())
            for ray, next_ray in zip(rays_record[:-1], rays_record[1:]):
                _plot_rays_3d(ax, ray.o, next_ray.o, out_ray.valid, colors, lss)

            import matplotlib.lines
            color_lines = [matplotlib.lines.Line2D([], [], color=c, linewidth=0.75) for c in colors]
            color_labels = [fr'${base.convert(_wl, "m", "um"):.5g}\mu m$' for _wl in wl.tolist()]
            fov_lines = [matplotlib.lines.Line2D([], [], color='black', linestyle=ls, linewidth=0.75) for ls in lss]
            fov_labels = [fr'${math.degrees(_fov):.5g}^\circ$' for _fov in fov.tolist()]
            ax.legend(color_lines + fov_lines, color_labels + fov_labels)

else:
    class CoaxialRayTracingVisMixIn:
        _mpl_err_msg = f'matplotlib required but not installed'

        def plot_cross_section(self, *args, **kwargs):
            raise RuntimeError(self._mpl_err_msg)

        def plot_spot_diagram(self, *args, **kwargs):
            raise RuntimeError(self._mpl_err_msg)

        def plot_psf_map(self, *args, **kwargs):
            raise RuntimeError(self._mpl_err_msg)


class CoaxialRayTracing(
    system.PsfImagingOptics,
    CoaxialRayTracingVisMixIn,
):
    """
    A class of sequential and ray-tracing-based optical system model.

    See :class:`~dnois.optics.PsfImagingOptics` for descriptions of more parameters.

    :param CoaxialSurfaceList surfaces: Surface list object.
    :param str psf_type: The way to calculate PSF. Default: ``inc_rect``.

        ``inc_rect``
            TODO

        ``inc_gaussian``
            TODO

        ``coh_kirchoff``
            TODO

        ``coh_huygens``
            TODO

        ``coh_fraunhofer``
            TODO
    :param str psf_center: The way to determine centers of computed PSFs.

        ``'linear'``
            PSFs are centered around ideal image points thus realistic distortion is simulated.

        ``'mean'``
            PSFs are centered around their "center of gravity".

        ``'chief'``
            PSFs are centered around the intersections of corresponding chief rays and image plane.
    :param str fov_type: The way to determine range of FoV.

        ``'perspective'``
            Determined by perspective relation i.e. size of sensor and :attr:`.perspective_focal_length`.

        ``'chief'``
            Determined by reversely tracing chief rays from edge of sensor to object space.

        ``'average'``
            Determined by averaging directions of rays traced from edge of sensor to object space.
    :param Callable sampler: A callable object whose signature is described by
        :meth:`dnois.optics.rt.Aperture.sampler`. This is typically created by this method as well.
    :param int coherent_tracing_samples: Number of samples in two directions
        for coherent tracing. Default: 512.
    :param str coherent_tracing_sampling_pattern: Sampling pattern for coherent tracing.
        Default: ``'quadrapolar'``.
    :param kwargs: Additional keyword arguments passed to :class:`PsfImagingOptics`.
    """
    _inherent = system.PsfImagingOptics._inherent + ['surfaces']
    _external = system.PsfImagingOptics._external + [
        'psf_type', 'psf_center',
        'coherent_tracing_samples', 'coherent_tracing_sampling_pattern',
        'fov_type', 'sampler'
    ]

    def __init__(
        self,
        surfaces: surf.CoaxialSurfaceList,
        sensor: Sensor = None,
        perspective_focal_length: float = None,
        psf_type: PsfType = 'inc_rect',
        psf_center: PsfCenter = 'linear',
        fov_type: FovType = 'perspective',
        sampler: surf.Sampler = None,
        coherent_tracing_samples: int = 512,
        coherent_tracing_sampling_pattern: str = 'quadrapolar',
        **kwargs
    ):
        super().__init__(sensor, perspective_focal_length, **kwargs)
        self.surfaces: surf.CoaxialSurfaceList = surfaces  #: Surface list.
        self.psf_type: PsfType = psf_type  #: See :class:`CoaxialRayTracing`.
        self.psf_center: PsfCenter = psf_center  #: See :class:`CoaxialRayTracing`.
        self.fov_type: FovType = fov_type  #: See :class:`CoaxialRayTracing`.
        self.sampler: surf.Sampler = sampler  #: See :class:`CoaxialRayTracing`.
        #: See :class:`CoaxialRayTracing`.
        self.coherent_tracing_samples: int = coherent_tracing_samples
        #: See :class:`CoaxialRayTracing`.
        self.coherent_tracing_sampling_pattern: str = coherent_tracing_sampling_pattern

        if self.sampler is None and len(self.surfaces) > 0:
            self.sampler = self.surfaces.first.aperture.sampler('random', 256)

    @utils.with_external
    def pointwise_render(
        self,
        scene: _sc.ImageScene,
        direct: bool = True,
        wl: Vector = None,
        depth: Vector | tuple[Ts, Ts] = None,
        psf_size: Size2d = None,
        norm_psf: bool = False,
        psf_type: PsfType = None,
        psf_center: PsfCenter = None,
        **kwargs,
    ) -> Ts:
        self._check_image_scene(scene)
        if direct:
            return self._direct_pw_render(scene, wl, depth, **kwargs)
        else:
            return super().pointwise_render(
                scene, wl, depth, psf_size, norm_psf,
                psf_center=psf_center, psf_type=psf_type, **kwargs
            )

    def cam2lens_z(self, depth: float | Ts) -> Ts:
        """
        Converts z-coordinates in :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`
        (i.e. depth) to those in :ref:`lens' coordinate system <guide_optics_rt_lcs>`.

        .. seealso::
            This is the inverse of :meth:`.len2cam_z`.

        :param depth: Depth.
        :type depth: float | Tensor
        :return: Z-coordinate in lens system. If ``depth`` is a float, returns a 0D tensor.
        :rtype: Tensor
        """
        if not torch.is_tensor(depth):
            depth = self.new_tensor(depth)
        return self.principal1 - depth

    def len2cam_z(self, z: float | Ts) -> Ts:
        """
        Converts z-coordinates in :ref:`lens' coordinate system <guide_optics_rt_lcs>`
        to those in :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>` (i.e. depth).

        .. seealso::
            This is the inverse of :meth:`.cam2lens_z`.

        :param z: Z-coordinate in lens' coordinate system.
        :type z: float | Tensor
        :return: Depth. If ``z`` is a float, returns a 0D tensor.
        :rtype: Tensor
        """
        if not torch.is_tensor(z):
            z = self.new_tensor(z)
        return self.principal1 - z

    def lens2cam(self, point: Ts) -> Ts:
        """
        Converts coordinates in :ref:`lens' coordinate system <guide_optics_rt_lcs>` to those in
        :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`.

        :param Tensor point: Coordinates in lens' coordinate system. A tensor with shape ``(..., 3)``.
        :return: Coordinates in camera's coordinate system. A tensor with shape ``(..., 3)``.
        :rtype: Tensor
        """
        _t.check_3d_vector(point, f'point in {self.lens2cam.__qualname__}')

        return torch.stack([-point[..., 0], point[..., 1], self.len2cam_z(point[..., 2])], -1)

    def cam2lens(self, point: Ts) -> Ts:
        """
        Converts coordinates in :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`
        into coordinates in :ref:`lens' coordinate system <guide_optics_rt_lcs>`.

        :param Tensor point: Coordinates in camera's coordinate system. A tensor with shape ``(..., 3)``.
        :return: Coordinates in lens' coordinate system. A tensor of shape ``(..., 3)``.
        :rtype: Tensor
        """
        _t.check_3d_vector(point, f'point in {self.cam2lens.__qualname__}')

        return torch.stack([-point[..., 0], point[..., 1], self.cam2lens_z(point[..., 2])], -1)

    def obj_proj_lens(self, point: Ts) -> Ts:
        """
        Returns x and y coordinates in :ref:`lens' coordinate system <guide_optics_rt_lcs>` of perspective projections
        of points in :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`
        ``point``. They can be viewed as ideal image points of object points ``point``.

        :param Tensor point: Points in camera's coordinate system, a tensor with shape ``(..., 3)``.
            It complies with :ref:`guide_imodel_ccs_inf`.
        :return: x and y coordinate of projected points, a tensor of shape ``(..., 2)``.
        :rtype: Tensor
        """
        xy_on_sensor = self.obj2tanfov(point) * self.reference.fl
        xy_on_sensor[..., 0] = -xy_on_sensor[..., 0]
        return xy_on_sensor

    def trace_ray(self, ray: BatchedRay) -> BatchedRay:
        """
        Trace a group of rays through surfaces until the image plane.
        If you want to trace rays until the last surface, call ``self.surfaces(ray)``.

        :param BatchedRay ray: Rays to trace.
        :return: Rays after tracing. Their origins are located at the image plane.
        :rtype: BatchedRay
        """
        out_ray: BatchedRay = self.surfaces(ray)
        ref_idx = self.surfaces.last.material.n(out_ray.wl, 'm')
        out_ray = out_ray.march_to(self.surfaces.total_length, ref_idx)
        return out_ray

    @utils.with_external
    def trace_point(self, point: Ts, wl: Vector = None, sampler: surf.Sampler = None) -> BatchedRay:
        """
        Trace a group of rays emitted from ``point`` through surfaces until the image plane.

        :param Tensor point: Coordinate of points in
            :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`.
            A tensor of shape ``(..., 3)``.
        :param wl: Wavelengths of rays. A float, a sequence of float or a tensor of shape ``(N_wl,)``.
        :param Callable sampler: A callable object whose signature is described by
            :meth:`dnois.optics.rt.Aperture.sampler`. This is typically created by this method as well.
        :return: Rays after tracing. Their origins are located at the image plane.
        :rtype: BatchedRay
        """
        sampler = self.pick('sampler', sampler)  # sampler is not an external parameter
        sampled = self.surfaces.first.sample(sampler)  # N_spp x 3
        d, _ = _make_direction(sampled, self.cam2lens(point).unsqueeze(-2))  # ... x N_spp|1 x 3
        ray = BatchedRay(sampled, d.unsqueeze(-3), wl.unsqueeze(-1))  # ... x N_wl x N_spp x 3
        out_ray = self.trace_ray(ray)  # ... x N_wl x N_spp
        return out_ray

    @torch.no_grad()
    def focus_to_(self, depth: Scalar) -> Self:
        """
        .. warning::

            This method is subject to change.
        """
        depth = typing.scalar(depth, dtype=self.dtype, device=self.device)
        z = self.cam2lens_z(depth)
        o = torch.stack((torch.zeros_like(z), torch.zeros_like(z), z))  # 3
        points = self.surfaces.first.sample(self.sampler)  # N_spp x 3
        d, _ = _make_direction(points, o)
        wl = self.wl.reshape(-1, 1)
        ray = BatchedRay(points, d, wl)  # N_wl x N_spp

        out_ray = self.trace_ray(ray)

        # solve marching distance by least square
        t = -(out_ray.x * out_ray.d_x + out_ray.y * out_ray.d_y)
        t = t / (out_ray.d_x.square() + out_ray.d_y.square())
        new_z = out_ray.z + t * out_ray.d_z
        new_z = new_z[out_ray.valid & new_z.isnan().logical_not()].mean()
        move = new_z - self.surfaces.total_length
        self.surfaces.last.distance.data += move

        return self

    @utils.with_external
    def psf(
        self,
        origins: Ts,
        psf_size: Size2d = None,
        wl: Vector = None,
        norm_psf: bool = None,
        psf_type: PsfType = None,
        psf_center: PsfCenter = None,
        **kwargs
    ) -> Ts:
        psf_center: PsfCenter  # eliminate None for type hint
        _t.check_3d_vector(origins, f'origins in {self.psf.__qualname__}')

        if psf_type == 'inc_rect':
            psf = self._psf_inc_rect(origins, psf_size, wl, psf_center, **kwargs)
        elif psf_type == 'inc_gaussian':
            psf = self._psf_inc_gaussian(origins, psf_size, wl, psf_center, **kwargs)
        elif psf_type == 'coh_huygens':
            psf = self._psf_coherent(origins, psf_size, wl, psf_center, False)
        elif psf_type == 'coh_kirchoff':
            psf = self._psf_coherent(origins, psf_size, wl, psf_center, True)
        elif psf_type == 'coh_fraunhofer':
            psf = self._psf_from_wavefront(origins, psf_size, wl, psf_center, **kwargs)
        else:
            raise ValueError(f'Unknown PSF type: {psf_type}')

        if norm_psf:
            psf = _func.norm_psf(psf)
        return psf

    def find_stop(
        self,
        depth: Scalar = float('inf'),
        ref_wl: Scalar = system.DEFAULT_WL,
        samples: int = 1024
    ) -> int:
        depth = typing.scalar(depth, self.dtype, self.device)
        ref_wl = typing.scalar(ref_wl, self.dtype, self.device)

        self._check_circ_aperture()
        self._check_circ_surf()

        # sample points on x-axis
        points = self.surfaces.first.sample('diameter', n=samples * 2)  # 2N x 3
        points = points[samples:]  # N x 3
        z = self.cam2lens_z(depth).item()
        origin = self.new_tensor([0, 0, z])  # 3
        d, _ = _make_direction(points, origin)  # N x 3
        ray = BatchedRay(points, d, ref_wl)  # N

        x_record = []
        for s in self.surfaces:
            ray = s(ray)
            x_record.append(ray.x)
        valid = ray.valid  # N
        stop_idx = None
        max_ratio = 0.
        for i, x in enumerate(x_record):
            valid_x = x[valid]
            ratio = valid_x.max() / self.surfaces[i].aperture.radius
            if ratio.item() > max_ratio:
                stop_idx = i
                max_ratio = ratio.item()

        if stop_idx is None:
            raise RuntimeError(f'Fail to find a stop')
        return stop_idx

    def entr_pupil(
        self,
        pupil_type: PupilType = 'paraxial',
        wl: Vector = None,
        wl_reduction: WlReduction = 'none',
        **kwargs
    ) -> PupilSpec:
        return self._pupil(True, pupil_type, wl, wl_reduction, **kwargs)

    def exit_pupil(
        self,
        pupil_type: PupilType = 'paraxial',
        wl: Vector = None,
        wl_reduction: WlReduction = 'none',
        **kwargs
    ) -> PupilSpec:
        return self._pupil(False, pupil_type, wl, wl_reduction, **kwargs)

    @utils.with_external
    def pupil_probe(self, entr: bool, ref_point: Ts, wl: Vector = None) -> PupilSpec:
        pass

    @utils.with_external(exclude='sampler')
    def pupil_trace(self, entr: bool, wl: Vector = None, sampler: surf.Sampler = None) -> PupilSpec:
        point, sublist = self._pupil_prepare(entr)

        stop_idx = self.surfaces.stop_idx  # this must exist which has been ensured in _pupil_prepare
        if (entr and stop_idx == 0) or (not entr and stop_idx == len(self.surfaces) - 1):
            return point[1], point[2]  # 0d

        adjacent = self.surfaces[stop_idx - 1] if entr else self.surfaces[stop_idx + 1]
        if sampler is None:
            end_points = adjacent.sample('unipolar')  # N_spp x 3
        else:
            end_points = adjacent.sample(sampler)  # N_spp x 3
        d = end_points - point  # N_spp x 3
        ray = BatchedRay(end_points, d, wl.unsqueeze(-1))  # N_wl x N_spp

        for s in sublist:
            ray = s(ray, not entr)
        focus_point = ray.focus(1)  # N_wl
        r, z = focus_point[1], focus_point[2]
        return r, z

    @utils.with_external
    def pupil_paraxial(self, entr: bool, wl: Vector = None) -> PupilSpec:
        point, sublist = self._pupil_prepare(entr)

        for s in sublist:
            if not isinstance(s, surf.ParaxialMixIn):
                raise RuntimeError(f'Paraxial behavior of surface {s.ctx.index} ({type(s).__name__}) is not defined')
            point = s.px_image_point(wl, point, not entr)  # (N_wl x )3
        r, z = point[..., 1], point[..., 2]  # N_wl or 0d
        return r, z

    # def entrance_pupil_probe(self, origin: Ts, wl: typing.Vector = None) -> tuple[Ts, Ts]:
    #     """
    #     .. attention::
    #
    #         This method applies when the apertures of all surfaces are circular.
    #
    #     .. warning::
    #
    #         This method is subject to change.
    #
    #     Finds the diameter and z-value of the entrance pupil using *probe* method.
    #
    #     :param Tensor origin: Origin of probe rays, a tensor with shape ``(..., 3)``.
    #         Its coordinate is defined in :ref:`lens' coordinate system <guide_optics_rt_lcs>`.
    #     :param wl: Wavelengths.
    #     :type wl: float | Sequence[float] | Tensor
    #     :return: Diameter and z-value of the entrance pupil, a pair of tensors of shape ``(..., N_wl)``.
    #     :rtype: tuple[Tensor, Tensor]
    #     """
    #     wl = self.pick('wl', wl)
    #
    #     self._check_circular()
    #     _t.check_3d_vector(origin, f'origin in {self.entrance_pupil_probe.__qualname__}')
    #     wl = wl.unsqueeze(-1)
    #
    #     origin = origin.unsqueeze(-2).unsqueeze(-3)  # ... x 1 x 1 x 3
    #     d_parallel, _ = _make_direction(
    #         torch.cat([self.new_tensor([0., 0.]), self.principal1.view(1)]), origin, True
    #     )  # ... x 1 x 1 x 3
    #
    #     r = self.surfaces.first.aperture.radius  # scalar
    #     edge_h = self.surfaces.first.h_extended(torch.zeros_like(r), r)  # scalar
    #     r = r + torch.sqrt(d_parallel[..., 2].reciprocal().square() - 1) * edge_h  # ... x 1 x 1
    #     axis_tmp = torch.linspace(-1, 1, find_chief_samples, device=self.device, dtype=self.dtype)
    #     x, y = torch.meshgrid(axis_tmp, axis_tmp, indexing='ij')
    #     x, y = x.flatten(), y.flatten()  # N_spp
    #     points_pre = torch.stack([x, y, torch.zeros_like(x)], dim=-1)  # N_spp x 3
    #     points_pre = points_pre * r.unsqueeze(-1)  # ... x 1 x N_spp x 3
    #     ray = BatchedRay(points_pre, d_parallel, wl)  # ... x N_wl x N_spp
    #     out_ray = self.surfaces(ray)
    #
    #     valid = out_ray.valid.broadcast_to(out_ray.shape)  # ... x N_wl x N_spp
    #     xy_valid = ray.o[..., :2].masked_fill(~valid.unsqueeze(-1), float('nan'))
    #     xy_mean = xy_valid.nanmean(-2, True)  # ... x N_wl x 1 x 2
    #     points_chief = torch.cat([xy_mean, torch.zeros_like(xy_mean[..., [0]])], -1)  # ... x N_wl x 1 x 3
    #     d_chief, l0_chief = _make_direction(points_chief, origin, True)  # ... x 1 x 1( x 3)
    #     chief_ray = BatchedRay(points_chief, d_chief, wl, 0.)  # ... x N_wl x 1

    # This method is adapted from
    # https://github.com/TanGeeGo/ImagingSimulation/blob/master/PSF_generation/ray_tracing/difftrace/analysis.py
    def wavefront_map(
        self,
        origin: Ts,
        wl: Vector = None,
        coherent_tracing_samples: int = DEFAULT_SAMPLES,
        coherent_tracing_sampling_pattern: str = 'quadrapolar',
    ) -> tuple[BatchedRay, Ts]:
        wl = self.pick('wl', wl)
        samples = self.pick('coherent_tracing_samples', coherent_tracing_samples)
        sampling_pattern = self.pick('coherent_tracing_sampling_pattern', coherent_tracing_sampling_pattern)

        chief_ray, ray, rs_roc, exit_pupil_distance = self._trace_opl_with_chief(
            origin, wl, samples, sampling_pattern
        )
        ref_idx = self.surfaces.mt_tail.n(ray.wl, 'm')
        opd = chief_ray.march(-rs_roc, ref_idx).opl - ray.opl  # ... x N_wl x N_spp
        opd[~ray.valid] = float('nan')
        return ray, opd / wl.unsqueeze(-1)  # ... x N_wl x N_spp

    @utils.with_external
    def chief_ray(
        self,
        point: Ts,
        wl: Vector = None,
        side: ChiefSide = 'obj',
        **kwargs
    ) -> BatchedRay:
        """

        :param point: coordinates in LCS
        :param wl:
        :param side:
        :param kwargs:
        :return:
        """
        if side == 'obj' or side == 'object':
            _, ap_z = self.entr_pupil(wl=wl, **kwargs)
        elif side == 'img' or side == 'image':
            _, ap_z = self.exit_pupil(wl=wl, **kwargs)
        else:
            raise ValueError(f'Side of chief ray must be obj, object, img or image, but got {side}')
        zero = torch.zeros_like(ap_z)
        chief_point = torch.stack([zero, zero, ap_z])  # 3
        d, _ = _make_direction(chief_point, point)  # ... x 3
        chief = BatchedRay(chief_point, d.unsqueeze(-2), wl)  # ... x N_wl
        return chief

    # Optical parameters
    # =============================

    @property
    def fov_x_lower(self) -> float:
        if self.fov_type == 'perspective':
            return super().fov_x_lower
        elif self.fov_type == 'chief':
            raise NotImplementedError()
        elif self.fov_type == 'average':
            raise NotImplementedError()
        else:
            raise ValueError(f'Unknown FoV type: {self.fov_type}')

    @property
    def fov_x_upper(self) -> float:
        if self.fov_type == 'perspective':
            return super().fov_x_upper
        elif self.fov_type == 'chief':
            raise NotImplementedError()
        elif self.fov_type == 'average':
            raise NotImplementedError()
        else:
            raise ValueError(f'Unknown FoV type: {self.fov_type}')

    @property
    def fov_y_lower(self) -> float:
        if self.fov_type == 'perspective':
            return super().fov_y_lower
        elif self.fov_type == 'chief':
            raise NotImplementedError()
        elif self.fov_type == 'average':
            raise NotImplementedError()
        else:
            raise ValueError(f'Unknown FoV type: {self.fov_type}')

    @property
    def fov_y_upper(self) -> float:
        if self.fov_type == 'perspective':
            return super().fov_y_upper
        elif self.fov_type == 'chief':
            raise NotImplementedError()
        elif self.fov_type == 'average':
            raise NotImplementedError()
        else:
            raise ValueError(f'Unknown FoV type: {self.fov_type}')

    @property
    def focal1(self) -> Ts:
        raise NotImplementedError()

    @property
    def focal2(self) -> Ts:
        raise NotImplementedError()

    @property
    def principal1(self) -> Ts:
        # TODO: currently depth=0 plane is assumed to be z=0 plane, while incorrect
        return self.new_tensor(0.)

    @property
    def principal2(self) -> Ts:
        raise NotImplementedError()

    def _check_circ_aperture(self):
        for s in self.surfaces:
            if not isinstance(s.aperture, surf.CircularAperture):
                s: surf.Surface
                raise NotImplementedError(
                    f'A function called requires all the surfaces have circular apertures, '
                    f'which is not satisfied for surface {s.ctx.index}'
                )

    def _check_circ_surf(self):
        for s in self.surfaces:
            if not isinstance(s, surf.CircularSurface):
                s: surf.Surface
                raise NotImplementedError(
                    f'A function called requires all the surfaces to be circularly symmetric, '
                    f'which is not satisfied for surface {s.ctx.index}'
                )

    # Serialization
    # ===========================
    @staticmethod
    def _todict_sampler(*arg, **kwargs):
        return None  # TODO: do not store sampler at present

    @classmethod
    def _pre_from_dict(cls, d: dict):
        d = super()._pre_from_dict(d)
        d['surfaces'] = surf.CoaxialSurfaceList.from_dict(d['surfaces'])
        return d

    # protected
    # ========================

    # This method is adapted from
    # https://github.com/TanGeeGo/ImagingSimulation/blob/master/PSF_generation/ray_tracing/difftrace/analysis.py
    @torch.no_grad()
    def _generate_rays(
        self,
        origin: Ts,
        wl: Ts,
        samples: int,
        sampling_pattern: str = 'quadrapolar',
        find_chief_samples: int = DEFAULT_FIND_CHIEF_SAMPLES,
    ) -> tuple[BatchedRay, BatchedRay]:
        """
        .. warning::

            This method is subject to change.
        """
        # origin is in lens' coordinate system
        if not isinstance(self.surfaces.first.aperture, surf.CircularAperture):
            raise NotImplementedError()
        _t.check_3d_vector(origin, f'origin in {self._generate_rays.__qualname__}')
        wl = wl.unsqueeze(-1)

        origin = origin.unsqueeze(-2).unsqueeze(-3)  # ... x 1 x 1 x 3
        d_parallel, _ = _make_direction(
            torch.cat([self.new_tensor([0., 0.]), self.principal1.view(1)]), origin, True
        )  # ... x 1 x 1 x 3

        r = self.surfaces.first.aperture.radius  # scalar
        edge_h = self.surfaces.first.h_extended(torch.zeros_like(r), r)  # scalar
        r = r + torch.sqrt(d_parallel[..., 2].reciprocal().square() - 1) * edge_h  # ... x 1 x 1
        axis_tmp = torch.linspace(-1, 1, find_chief_samples, device=self.device, dtype=self.dtype)
        x, y = torch.meshgrid(axis_tmp, axis_tmp, indexing='ij')
        x, y = x.flatten(), y.flatten()  # N_spp
        points_pre = torch.stack([x, y, torch.zeros_like(x)], dim=-1)  # N_spp x 3
        points_pre = points_pre * r.unsqueeze(-1)  # ... x 1 x N_spp x 3
        ray = BatchedRay(points_pre, d_parallel, wl, d_normed=True)  # ... x N_wl x N_spp
        out_ray = self.surfaces(ray)

        valid = out_ray.valid.broadcast_to(out_ray.shape)  # ... x N_wl x N_spp
        xy_valid = ray.o[..., :2].masked_fill(~valid.unsqueeze(-1), float('nan'))
        xy_mean = xy_valid.nanmean(-2, True)  # ... x N_wl x 1 x 2
        points_chief = torch.cat([xy_mean, torch.zeros_like(xy_mean[..., [0]])], -1)  # ... x N_wl x 1 x 3
        d_chief, l0_chief = _make_direction(points_chief, origin)  # ... x 1 x 1( x 3)
        chief_ray = BatchedRay(points_chief, d_chief, wl, 0.)  # ... x N_wl x 1

        # mimicking np.nanmax and np.nanmin
        xy_min = xy_valid.nan_to_num(nan=float('inf')).amin(-2, True)
        xy_max = xy_valid.nan_to_num(nan=-float('inf')).amax(-2, True)
        xy_shift_min = torch.abs(xy_min - xy_mean).unsqueeze(-2)  # ... x N_wl x 1 x 1 x 2
        xy_shift_max = torch.abs(xy_max - xy_mean).unsqueeze(-2)  # ... x N_wl x 1 x 1 x 2

        axis = torch.linspace(-1, 1, samples, dtype=self.dtype, device=self.device)
        if sampling_pattern == 'quadrapolar':
            h_p, w_p = torch.meshgrid(-axis, axis, indexing='ij')
            theta = torch.arctan2(h_p, w_p)
            o_p = torch.stack((theta.cos(), theta.sin()), dim=-1)
            o_p *= torch.max(h_p.abs(), w_p.abs()).unsqueeze(-1)  # samples x samples x 2
        elif sampling_pattern == 'rect':
            o_p = torch.stack(torch.meshgrid(axis, -axis, indexing='xy'), -1)
        else:
            raise ValueError(f'Unknown sampling pattern for {self._generate_rays.__qualname__}: {sampling_pattern}')

        o_shape = xy_valid.shape[:-2] + (samples, samples, 3)
        o = torch.zeros(o_shape, dtype=self.dtype, device=self.device)
        o[..., :samples // 2, :, 1] = o_p[:samples // 2, :, 1] * xy_shift_max[..., 1]
        o[..., samples // 2:, :, 1] = o_p[samples // 2:, :, 1] * xy_shift_min[..., 1]
        o[..., :, :samples // 2, 0] = o_p[:, :samples // 2, 0] * xy_shift_max[..., 0]
        o[..., :, samples // 2:, 0] = o_p[:, samples // 2:, 0] * xy_shift_min[..., 0]
        o[..., :2] += xy_mean.unsqueeze(-2)
        points_sample = o.flatten(-3, -2)
        d_sample, l0 = _make_direction(points_sample, origin, True)  # ... x 1 x N_spp'( x 3)
        # to reduce magnitude of opl and subsequently floating point error
        l0 = l0 - l0_chief  # ... x 1 x N_spp'
        if origin[..., 2].isinf().any():
            l0 = torch.where(
                origin[..., 2].isinf(),  # ... x 1 x 1
                torch.sum((points_sample - points_chief) * d_parallel, -1),
                l0
            )  # ... x N_wl x N_spp'
        ref_idx = self.surfaces.mt_head.n(wl, 'm')
        # ... x N_wl x N_spp'
        ray = BatchedRay(points_sample, d_sample, wl, l0 * ref_idx, d_normed=True)
        return ray, chief_ray

    def _trace_opl_with_chief(
        self,
        origin: Ts,
        wl: Vector,
        samples: int = DEFAULT_SAMPLES,
        sampling_pattern: str = 'quadrapolar',
    ) -> tuple[BatchedRay, BatchedRay, Ts, Ts]:
        # ... x N_wl x N_spp(1)
        ray, chief_ray = self._generate_rays(self.cam2lens(origin), wl, samples, sampling_pattern)

        ray = self.surfaces(ray)
        chief_ray = self.trace_ray(chief_ray)
        radial_offset = torch.sqrt(chief_ray.o[..., :2].square().sum(-1))
        d_proj = torch.sqrt(chief_ray.d[..., :2].square().sum(-1))
        rs_roc = radial_offset / d_proj  # ... x N_wl x 1
        exit_pupil_distance = torch.sqrt(rs_roc.square() - radial_offset.square()).squeeze(-1)  # ... x N_wl

        shift = chief_ray.o - ray.o
        dp = torch.sum(shift * ray.d, dim=-1)  # dot product
        length2rs = dp - torch.sqrt(dp.square() - shift.square().sum(-1) + rs_roc.square())
        ref_idx = self.surfaces.mt_tail.n(ray.wl, 'm')
        ray.march_(length2rs, ref_idx)
        return chief_ray, ray, rs_roc, exit_pupil_distance  # ... x N_wl x N_spp

    def _direct_pw_render(
        self,
        scene: _sc.ImageScene,
        wl: Ts,
        depth: Ts | tuple[Ts, Ts],
        sampler: surf.Sampler,
        vignette: bool = True,
    ) -> Ts:
        self._check_scene(scene)
        if wl.numel() != scene.n_wl:
            raise ValueError(f'A scene with {self.wl.numel()} wavelengths expected, got {scene.n_wl}')
        if len(self.surfaces) == 0:
            raise RuntimeError(f'No surface available')

        scene = scene.batch()
        n_b, n_wl, n_h, n_w = scene.image.shape

        depth_map = self._make_depth_map(scene, depth)  # B x H x W
        o = self.points_grid((n_h, n_w), depth_map, True)  # B x H x W x 3
        o = self.cam2lens(o)  # B x H x W x 3

        points = self.surfaces.first.sample(sampler)  # N_spp x 3
        spp = points.size(0)
        o = o.unsqueeze(-2).unsqueeze(-5)
        d, _ = _make_direction(points, o)  # B x 1 x H x W x N_spp x 3

        wl = wl.view(1, -1, 1, 1, 1)
        ray = BatchedRay(points, d, wl)  # B x N_wl x H x W x N_spp

        out_ray = self.trace_ray(ray)

        x, y = -out_ray.x / self.sensor.pixel_size[1], out_ray.y / self.sensor.pixel_size[0]
        x, y = x + self.sensor.pixel_num[1] / 2, y + self.sensor.pixel_num[0] / 2
        c_a, r_a = torch.floor(x.detach() + 0.5).long(), torch.floor(y.detach() + 0.5).long()
        in_region = (c_a >= 0) & (c_a <= n_w) & (r_a >= 0) & (r_a <= n_h)  # B x N_wl x H x W x N_spp
        c_a[~in_region] = 0
        r_a[~in_region] = 0
        c_as, r_as = c_a - 1, r_a - 1
        w_c, w_r = c_a - x + 0.5, r_a - y + 0.5
        iw_c, iw_r = 1 - w_c, 1 - w_r

        b_wl_idx = (
            torch.arange(n_b, device=self.device).view(-1, 1, 1, 1, 1),
            torch.arange(n_wl, device=self.device).view(1, -1, 1, 1, 1),
        )
        image = self.new_zeros((n_b, n_wl, n_h + 2, n_w + 2))
        mask = out_ray.valid & in_region
        gt_image = scene.image.unsqueeze(-1)
        if vignette:
            gt_image = gt_image * spp / (out_ray.valid.sum(dim=-1, keepdim=True) + 1e-5)
        _gt1 = gt_image * w_c
        _gt2 = gt_image * iw_c
        image.index_put_(b_wl_idx + (r_as, c_as), torch.where(mask, _gt1 * w_r, 0), True)  # top left
        image.index_put_(b_wl_idx + (r_a, c_as), torch.where(mask, _gt1 * iw_r, 0), True)  # bottom left
        image.index_put_(b_wl_idx + (r_as, c_a), torch.where(mask, _gt2 * w_r, 0), True)  # top right
        image.index_put_(b_wl_idx + (r_a, c_a), torch.where(mask, _gt2 * iw_r, 0), True)  # bottom right
        image = image[..., :-2, :-2] / spp
        return image

    def _find_xy_center(self, psf_center, origins, out_ray, wl):
        if psf_center == 'linear':
            xy_center = self.obj_proj_lens(origins)[..., None, None, :]  # ... x 1 x 1 x 2
        elif psf_center == 'mean':
            xy_center = out_ray.o[..., :2]  # ... x N_wl x N_spp x 2
            valid = out_ray.valid.unsqueeze(-1)  # ... x N_wl x N_spp x 1
            xy_center = torch.where(valid, xy_center, 0).sum(-2, True) / valid.sum(-2, True)  # ... x N_wl x 1 x 2
            xy_center = xy_center.mean(-3, True)  # ... x 1 x 1 x 2
        elif psf_center == 'chief':
            origins = self.cam2lens(origins)
            chief = self.chief_ray(origins, wl)  # ... x N_wl
            out_chief = self.trace_ray(chief)
            xy_center = out_chief.o[..., None, :2]  # ... x N_wl x 1 x 2
            xy_center = xy_center.mean(-3, True)  # ... x 1 x 1 x 2
        else:
            raise ValueError(f'Unsupported PSF center type for simple incoherent PSF: {psf_center}')
        return xy_center

    def _psf_inc_rect(
        self,
        origins: Ts,  # ... x 3
        psf_size: tuple[int, int],
        wl: Ts,  # N_wl
        psf_center: PsfCenter,
        sampler: surf.Sampler = None,
    ) -> Ts:
        sampler = self.pick('sampler', sampler)
        f_name = self._psf_inc_rect.__name__

        out_ray = self.trace_point(origins, wl, sampler)  # ... x N_wl x N_spp
        out_ray.update_valid_(~out_ray.x.isnan() & ~out_ray.y.isnan())  # TODO:?
        out_ray = self.capture_hook(f'{f_name}.out_ray', out_ray)
        n_spp = out_ray.shape[-1]

        xy_center = self._find_xy_center(psf_center, origins, out_ray, wl)
        xy = out_ray.o[..., :2]  # ... x N_wl x N_spp x 2
        # xy = torch.where(out_ray.valid.unsqueeze(-1), xy, 0)
        xy = xy - xy_center  # ... x N_wl x N_spp x 2
        if xy.requires_grad and ddb.debugging():
            # print(xy.numel(), xy.shape, out_ray.valid_percentage())
            xy.register_hook(ddb.grad_hook_check_peculiar(f'xy in {self._psf_inc_rect.__qualname__}'))

        xy = xy / self.new_tensor([self.sensor.pixel_size]).flip(0)
        xy = self.capture_hook(f'{f_name}.xy', xy)
        x, y = xy.unbind(-1)  # ... x N_wl x N_spp
        # if PSF size is odd, the center is N/2, relative positions are -N//2, ..., N//2
        # if PSF size is even, the center is (N+1)/2, relative positions are -N//2, ..., N//2-1
        x, y = x + (psf_size[1] // 2 + 0.5), y + (psf_size[0] // 2 + 0.5)
        c_a, r_a = torch.floor(x.detach() + 0.5).long(), torch.floor(y.detach() + 0.5).long()

        in_region = c_a.ge(0) & c_a.le(psf_size[1]) & r_a.ge(0) & r_a.le(psf_size[0])  # ... x N_wl x N_spp
        in_region = self.capture_hook(f'{f_name}.in_region', in_region)
        mask = out_ray.valid & in_region  # ... x N_wl x N_spp
        for t in (x, y, c_a, r_a):  # mask out invalid rays in these four tensors
            t[~mask] = 0

        c_as, r_as = c_a - 1, r_a - 1
        w_c, w_r = c_a - x + 0.5, r_a - y + 0.5
        iw_c, iw_r = 1 - w_c, 1 - w_r
        psf = self.new_zeros(out_ray.shape[:-1] + (psf_size[0] + 2, psf_size[1] + 2))  # ... x N_wl x (H+2) x (W+2)
        pre_idx = [
            _t.as1d(torch.arange(dim_size, device=self.device), mask.ndim, i)
            for i, dim_size in enumerate(mask.shape[:-1])
        ]

        psf.index_put_(pre_idx + [r_as, c_as], torch.where(mask, w_c * w_r, 0), True)  # top left
        psf.index_put_(pre_idx + [r_a, c_as], torch.where(mask, w_c * iw_r, 0), True)  # bottom left
        psf.index_put_(pre_idx + [r_as, c_a], torch.where(mask, iw_c * w_r, 0), True)  # top right
        psf.index_put_(pre_idx + [r_a, c_a], torch.where(mask, iw_c * iw_r, 0), True)  # bottom right

        psf = psf[..., :-2, :-2]  # ... x N_wl x H x W
        psf = psf.flip(-1)
        psf = psf / n_spp  # total energy of each ray is 1
        return psf

    def _psf_inc_gaussian(
        self,
        origins: Ts,  # ... x 3
        size: tuple[int, int],
        wl: Ts,  # N_wl
        psf_center: PsfCenter,
        sampler: surf.Sampler = None,
    ) -> Ts:
        sampler = self.pick('sampler', sampler)

        out_ray = self.trace_point(origins, wl, sampler)  # ... x N_wl x N_spp
        out_ray.update_valid_(~out_ray.x.isnan() & ~out_ray.y.isnan())  # TODO:?

        xy_center = self._find_xy_center(psf_center, self.cam2lens(origins), out_ray, wl)
        xy = out_ray.o[..., :2] - xy_center  # ... x N_wl x N_spp x 2
        py, px = self.sensor.pixel_size
        valid = xy[..., 0].abs().le(px * (size[1] / 2 + 5)) & xy[..., 1].abs().le(py * (size[0] / 2 + 5))
        valid.logical_and_(out_ray.valid)  # ... x N_wl x N_spp

        psf = self.new_empty(out_ray.shape[:-1] + size)  # ... x N_wl x H x W
        ry, rx = utils.grid(size, self.sensor.pixel_size, dtype=self.dtype, device=self.device)
        rxy = torch.stack([rx, ry], -1)  # H x W x 2
        pixel_diag = math.sqrt(px ** 2 + py ** 2)
        sigma = pixel_diag / 3
        a, b = 1 / (math.sqrt(2 * math.pi) * sigma), -1 / (2 * sigma * sigma)
        for i in range(psf.size(-2)):
            for j in range(psf.size(-1)):
                r2 = torch.square(rxy[i, j] - xy).sum(-1)  # ... x N_wl x N_spp
                w = a * torch.exp(b * r2)  # ... x N_wl x N_spp
                psf[..., i, j] = torch.where(valid, w, 0).sum(-1)  # ... x N_wl

        psf = psf.flip(-1)
        return psf

    # This method is adapted from
    # https://github.com/TanGeeGo/ImagingSimulation/blob/master/PSF_generation/ray_tracing/difftrace/analysis.py
    def _psf_coherent(
        self,
        origins: Ts,
        size: tuple[int, int],
        wl: Ts,
        psf_center: PsfCenter,
        oblique: bool,
        samples: int = DEFAULT_SAMPLES,
    ) -> Ts:
        """This method is subject to change."""
        chief_ray, ray, rs_roc, _ = self._trace_opl_with_chief(origins, wl, samples)

        lim = ((size[0] - 1) // 2, (size[1] - 1) // 2)
        # x and y should decrease when index gets large since:
        # x coordinate of PSF should be in camera's coordinate system
        # but the computation is performed in lens' coordinate system
        # so a horizontal flipping is needed
        # and large index for y means lower position i.e. small y
        y, x = [torch.linspace(
            lim[i], -lim[i], size[i], device=self.device, dtype=self.dtype
        ) * self.sensor.pixel_size[i] for i in range(2)]
        x, y = torch.meshgrid(x, y, indexing='xy')  # H x W

        if psf_center == 'linear':
            center_xy = self.obj_proj_lens(origins)[..., None, None, None, :]  # ... x 1 x 1 x 1 x 2
        elif psf_center == 'chief':
            center_xy = chief_ray.o[..., :2].unsqueeze(-2)  # ... x N_wl x 1 x 1 x 2
        else:
            raise ValueError(f'Unsupported PSF center type for coherent PSF: {psf_center}')

        sampling_grid = center_xy + torch.stack([x, y], -1)  # ... x N_wl x H x W x 2
        sampling_grid = torch.cat([
            sampling_grid, self.surfaces.total_length.broadcast_to(sampling_grid.shape[:-1]).unsqueeze(-1)
        ], -1)  # ... x N_wl x H x W x 3

        # ... x N_wl x H x W x N_spp x 3
        rs2grid_points = sampling_grid.unsqueeze(-2) - ray.o[..., None, None, :, :]
        r_proj = torch.sum(ray.d[..., None, None, :, :] * rs2grid_points, -1)
        wave_vec = base.wave_vec(wl.reshape(-1, 1, 1, 1))
        phase = (r_proj + ray.opl[..., None, None, :]) * wave_vec

        if oblique:
            r_unit = torch.linalg.vector_norm(rs2grid_points)  # ... x N_wl x H x W x N_spp x 3
            rs_normal = ray.o - chief_ray.o
            rs_normal = torch.linalg.vector_norm(rs_normal)  # ... x N_wl x N_spp x 3
            cosine_prop = torch.sum(rs_normal[..., None, None, :, :] * r_unit, -1)
            cosine_rs = torch.sum(rs_normal * ray.d, -1)  # N_fov x N_D x N_wl x N_spp
            obliquity = (cosine_rs[..., None, None, :] + cosine_prop) / 2
            field = torch.polar(obliquity, phase)
        else:
            field = _t.expi(phase)
        field = field.sum(-1)  # N_fov x N_D x N_wl x H x W
        psf = _t.abs2(field)
        return psf  # N_fov x N_D x N_wl x H x W

    def _psf_from_wavefront(
        self,
        origins: Ts,
        size: tuple[int, int],
        wl: Ts,
        psf_center: PsfCenter,
        samples: int = DEFAULT_SAMPLES,
    ) -> Ts:
        """This method is subject to change."""
        if psf_center != 'chief':
            raise ValueError(f'Unsupported PSF center type for wavefront PSF: {psf_center}')

        chief_ray, ray, rs_roc, exit_pupil_distance = self._trace_opl_with_chief(origins, wl, samples, 'rect')
        exit_pupil_distance = exit_pupil_distance.detach()  # TODO: bug to fix

        ref_idx = self.surfaces.mt_tail.n(ray.wl, 'm')
        opd = chief_ray.march(-rs_roc, ref_idx).opl - ray.opl  # ... x N_wl x N_spp
        opd[~ray.valid] = float('nan')
        phase = opd * base.wave_vec(wl.unsqueeze(-1))
        if phase.requires_grad:  # TODO: bug to fix
            phase.register_hook(lambda g: torch.nan_to_num(g, nan=0.))

        spp = phase.size(-1)
        grid_size = int(math.sqrt(spp))
        # ... x N_wl x samples x samples, in lens' coordinate system
        phase = phase.reshape(phase.shape[:-1] + (grid_size, grid_size))
        phase = phase.flip(-2)  # phase on exit pupil in camera's coordinate system

        # TODO: wfe_u and wfe_v are assumed to be uniform in current code, enabling direct bilinear interpolation
        # ... x N_wl x samples x samples
        wfe_u = ray.x.reshape(ray.x.shape[:-1] + (grid_size, grid_size))
        wfe_v = ray.y.reshape(ray.y.shape[:-1] + (grid_size, grid_size))
        u_mean, v_mean = wfe_u.nanmean(-2), wfe_v.nanmean(-1)  # ... x N_wl x samples
        # ... x N_wl
        du, dv = (u_mean.amax(-1) - u_mean.amin(-1)) / grid_size, (v_mean.amax(-1) - v_mean.amin(-1)) / grid_size
        scale = exit_pupil_distance * wl  # ... x N_wl

        # all: ... x N_wl
        factor_x = grid_size * du * self.sensor.pixel_size[1] / scale
        factor_y = grid_size * dv * self.sensor.pixel_size[0] / scale
        factor_x, factor_y = factor_x.max().ceil().int().item(), factor_y.max().ceil().int().item()
        range_u, range_v = factor_x * scale / self.sensor.pixel_size[1], factor_y * scale / self.sensor.pixel_size[0]
        new_u_num, new_v_num = range_u / du, range_v / dv
        new_u_num, new_v_num = new_u_num.mean().round().int().item(), new_v_num.mean().round().int().item()
        du2, dv2 = range_u / new_u_num, range_v / new_v_num

        # bilinear interpolation
        new_v, new_u = utils.grid(
            (new_v_num, new_u_num), (dv2, du2), symmetric=True, dtype=self.dtype, device=self.device,
        )  # ... x N_wl x MH x MW
        new_r = new_v / dv[..., None, None] + (grid_size - 1) / 2
        new_c = new_u / du[..., None, None] + (grid_size - 1) / 2
        upper_r, left_c = new_r.floor().int(), new_c.floor().int()
        lower_r, right_c = upper_r + 1, left_c + 1
        valid_r1, valid_r2 = upper_r.clamp(0, grid_size - 1), lower_r.clamp(0, grid_size - 1)
        valid_c1, valid_c2 = left_c.clamp(0, grid_size - 1), right_c.clamp(0, grid_size - 1)
        _r_vec = torch.stack([lower_r - new_r, new_r - upper_r], -1).unsqueeze(-2)  # ... x N_wl x MH x MW x 1 x 2
        _c_vec = torch.stack([right_c - new_c, new_c - left_c], -1).unsqueeze(-1)  # ... x N_wl x MH x MW x 2 x 1
        pre_idx = [torch.arange(dim_size, device=self.device) for dim_size in phase.shape[:-2]]
        pre_idx = [_t.as1d(idx, len(pre_idx) + 2, i) for i, idx in enumerate(pre_idx)]
        _mat = torch.stack([
            torch.stack([phase[*pre_idx, valid_r1, valid_c1], phase[*pre_idx, valid_r1, valid_c2]], -1),
            torch.stack([phase[*pre_idx, valid_r2, valid_c1], phase[*pre_idx, valid_r2, valid_c2]], -1),
        ], -2)  # ... x N_wl x MH x MW x 2 x 2
        interp_phase = _r_vec @ _mat @ _c_vec
        interp_phase = interp_phase.squeeze(-1).squeeze(-1)  # ... x N_wl x MH x MW
        interp_phase[
            (upper_r != valid_r1) | (lower_r != valid_r2) | (left_c != valid_c1) | (right_c != valid_c2)
            ] = float('nan')  # ... x N_wl x MH x MW

        ep_field = _t.expi(interp_phase)
        ep_field[interp_phase.isnan()] = 0.

        psf = _t.abs2(fourier.ft2(ep_field))  # ... x N_wl x samples x samples

        if factor_x == 1 and factor_y == 1:
            psf = utils.resize(psf, size)
        else:
            psf = utils.resize(psf, (size[0] * factor_y, size[1] * factor_x))
            slices = [[
                psf[..., i::factor_y, j::factor_x] for j in range(factor_x)
            ] for i in range(factor_y)]
            psf = sum([sum(slc) for slc in slices]) / (factor_x * factor_y)

        psf = psf.flip(-2)
        return psf

    @utils.with_external
    def _pupil(
        self,
        entr: bool,
        pupil_type: PupilType = 'paraxial',
        wl: Vector = None,
        wl_reduction: WlReduction = 'none',
        **kwargs
    ) -> PupilSpec:
        if pupil_type == 'paraxial':
            r, z = self.pupil_paraxial(entr, wl, **kwargs)
        elif pupil_type == 'probe':
            r, z = self.pupil_probe(entr, wl=wl, **kwargs)
        elif pupil_type == 'trace':
            r, z = self.pupil_trace(entr, wl, **kwargs)
        else:
            raise ValueError(f'Unknown pupil type: {pupil_type}, '
                             f'available: {", ".join(typing.get_args(PupilType))}')

        if r.ndim == z.ndim == 0 or wl_reduction == 'none':
            return r, z
        elif wl_reduction == 'mean':
            return r.mean(), z.mean()
        elif wl_reduction == 'center':
            return r[wl.size(0) // 2], z[wl.size(0) // 2]
        else:
            raise ValueError(f'Unknown wl_reduction: {wl_reduction}, '
                             f'available: {", ".join(typing.get_args(WlReduction))}')

    def _pupil_prepare(self, entr: bool) -> tuple[Ts, list[surf.Surface]]:
        self._check_circ_aperture()
        self._check_circ_surf()

        stop = self.surfaces.stop
        if stop is None:
            raise NotImplementedError(f'A stop must be specified to compute paraxial pupil currently')

        ap = typing.cast(surf.CircularAperture, stop.aperture)
        idx = stop.ctx.index
        z_stop = stop.ctx.baseline  # 0d
        r_stop = ap.radius  # 0d
        point = torch.stack([torch.zeros_like(r_stop), r_stop, z_stop])  # 3

        if entr:
            sublist = list(reversed(self.surfaces[:idx]))
        else:
            sublist = self.surfaces[idx + 1:]
        return point, sublist
