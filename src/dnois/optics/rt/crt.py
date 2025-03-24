import math
import warnings

import torch

from . import surf, rto
from .ray import BatchedRay
from .. import system, _func
from ... import scene as _sc, base, utils, fourier, torch as _t, ext
from ...base import typing
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
ImagingModel = typing.Literal['psf', 'forward_rt', 'backward_rt']


def _check_arg(arg: Any, n1: str, n2: str):
    if arg is not None:
        warnings.warn(f'{n1} and {n2} are given simultaneously, {n2} will be ignored')


def _plot_set_ax(ax, x_range: tuple[float, float]):
    x_length = x_range[1] - x_range[0]
    ax.set_xlim(x_range[0] - 0.05 * x_length, x_range[1] + 0.05 * x_length)
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
    # of x and y FoV, respectively
    # sampled_point and origin are all assumed to be in LCS
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
        _ls_surf = {'color': 'black', 'linewidth': 1}
        _ls_bold = {'color': 'black', 'linewidth': 2}

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
                    x = utils.t4plot(ray_out.x[j] - chief_ray_out.x[j])
                    y = utils.t4plot(ray_out.y[j] - chief_ray_out.y[j])
                    ax.scatter(
                        x, y,
                        s=2, c=utils.wl2rgb(wl[j].item(), output_format='hex'), label=f'{wl[j].item():.4g}',
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
            depth: Scalar = float('inf'),
            height: Vector = None,
            wl: Vector = None,
            init_rays: int = 20,
            legend: bool = True,
        ) -> tuple[plt.Figure, plt.Axes]:
            self._check_circ_aperture()
            self._check_circ_surf()

            if torch.is_tensor(depth) and depth.numel() > 1:
                raise RuntimeError('Cross section figure for multiple depths is not implemented yet')
            depth = typing.scalar(depth.squeeze(), device=self.device, dtype=self.dtype)
            if fig is None:
                fig, ax = plt.subplots(figsize=(12.8, 9.6), subplot_kw={'frameon': True})
            else:
                ax = fig.axes[0][0]
            if height is None:
                fov_half = self.reference.fov_half
                fovs = [0., fov_half * 0.5 ** 0.5, fov_half]
                fovs = [(0., fov) for fov in fovs]
                o = self.fovd2obj(fovs, depth)  # (3, 3)
                height = o[:, 1]  # (3,)
            else:
                height = typing.vector(height, device=self.device, dtype=self.dtype)
                o = torch.stack([
                    torch.zeros_like(height),
                    height,
                    torch.full_like(height, self.cam2lens_z(depth).item())
                ], -1)  # (N, 3)

            self._plot_components(ax)

            z_obj = self.cam2lens_z(depth).item()
            if z_obj != -float('inf'):
                max_h = height.abs().max().item()
                y = torch.linspace(-max_h, max_h, 100, device=self.device)
                ax.plot(utils.t4plot(torch.full_like(y, z_obj)), utils.t4plot(y), **self._ls_bold)

            # image_plane
            if self.sensor is not None:
                diag_length = (self.sensor.h ** 2 + self.sensor.w ** 2) ** 0.5
                sensor_z = self.surfaces.total_length.item()
                ax.plot([sensor_z, sensor_z], [-diag_length / 2, diag_length / 2], **self._ls_bold)

            # rays
            sampled = self.surfaces.first.sample('diameter', init_rays, torch.pi / 2)  # N_spp x 3
            o = o.unsqueeze(-2).unsqueeze(-2)  # N x 1 x 1 x 3
            d, _ = _make_direction(sampled, o, True)  # N x 1 x N_spp x 3
            if z_obj == -float('inf'):
                ray = BatchedRay(sampled, d, wl.reshape(1, -1, 1))  # N x N_wl x N_spp
            else:
                ray = BatchedRay(o, d, wl.reshape(1, -1, 1))  # N x N_wl x N_spp
            self._plot_rays(ax, ray, height, wl, legend)

            x_range = (0. if z_obj == -float('inf') else z_obj, self.surfaces.total_length.item())
            _plot_set_ax(ax, x_range)
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

        def _plot_rays(self: 'CoaxialRayTracing', ax, ray: BatchedRay, height: Ts, wl: Ts, legend: bool):
            # ray: N_fov x N_wl x N_spp
            isinf = self.depth.isinf().item()
            colors = [utils.wl2rgb(_wl, output_format='hex') for _wl in wl.tolist()]
            lss = _plot_linestyles(height.numel())

            if isinf:
                ray.broadcast_().march_to_(ray.new_tensor(0.))

            rays_record = [ray.broadcast_()]
            for sf in self.surfaces:
                out_ray = sf(ray)
                rays_record.append(out_ray.broadcast_())
                ray = out_ray
            out_ray = ray.march_to(self.surfaces.total_length)
            rays_record.append(out_ray.broadcast_())
            for ray, next_ray in zip(rays_record[:-1], rays_record[1:]):
                _plot_rays_3d(ax, ray.o, next_ray.o, out_ray.valid, colors, lss)

            if legend:
                import matplotlib.lines
                color_lines = [matplotlib.lines.Line2D([], [], color=c, linewidth=0.75) for c in colors]
                color_labels = [fr'${utils.fmt(base.convert(_wl, "m", "um"))}\mu m$' for _wl in wl.tolist()]
                fov_lines = [matplotlib.lines.Line2D([], [], color='black', linestyle=ls, linewidth=0.75) for ls in lss]
                if isinf:
                    fov_labels = [fr'${utils.fmt(math.degrees(math.atan(-h)))}^\circ$' for h in height.tolist()]
                else:
                    fov_labels = [fr'${utils.fmt(h)}{base.Length.default()}$' for h in height.tolist()]
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
    rto.ForwardRayTracingOptics,
    CoaxialRayTracingVisMixIn,
):
    """
    A class of sequential and ray-tracing-based optical system model.

    See :class:`~dnois.optics.PsfImagingOptics` for descriptions of more parameters.

    :param CoaxialSurfaceList surfaces: Surface list object.
    :param str imaging_model: The way to render imaged radiance field. Default: ``'psf'``.

        ``'psf'``
            Use PSF to render images. See :class:`~system.PsfImagingOptics` for more details.

        ``'forward_rt'``
            Rays emitting from all object points are traced and superposed on image plane simultaneously.
    :param str psf_type: The way to calculate PSF. Default: ``inc_rect``.

        ``'inc_rect'``
            Intensity distribution rays imparted on image plane are modeled as a rectangular
            with size identical to a sensor pixel and are superposed incoherently [#yang2023aberration]_.

        ``'inc_gaussian'``
            Intensity distribution rays imparted on image plane are modeled as a gaussian
            distribution and are superposed incoherently [#li2021end]_.

        ``'coh_kirchoff'``
            Intersection of each ray and exit pupil is considered as a secondary point source.
            The complex amplitude on image plane is determined as superposition of their wave
            according to Huygens-Fresnel Principle [#chen2021optical]_.

        ``'coh_huygens'``
            Similar to ``'coh_kirchoff'`` but without oblique factor.

        ``'coh_fraunhofer'``
            The complex amplitude on image plane is computed as Fraunhofer diffraction,
            i.e. Fourier transform of pupil function.
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
    :param str wl_reduction: The way to reduce wavelength dimension when some computation results
        depend on wavelength. Default: ``'mean'``.

        ``'none'``
            No reduction.

        ``'mean'``
            Reduce wavelength dimension by taking mean.

        ``'center'``
            Reduce wavelength dimension by taking center.
    :param str pupil_type: The way to determine entrance or exit pupil. Default: ``'paraxial'``.

        ``'probe'``
            Find pupil by calling :meth:`.pupil_probe`.

        ``'trace'``
            Find pupils by calling :meth:`.pupil_trace`.

        ``'paraxial'``
            Find pupils by calling :meth:`.pupil_paraxial`.
    :param int repetitions: Number of repetitions of computing in ``'forward_rt'`` mode.
        Typically, this mode requires an exceedingly
        huge amount of memory to compute, in which case one can set :attr:`.sampler` to a
        random sampler (see :meth:`~surf.Aperture.sampler`) with few sampling points,
        run rendering ``repetitions`` times and get their average to get rendered image
        with virtually many sampling points while memory footprint is reduced. Default: ``1``.
    :param kwargs: Additional keyword arguments passed to :class:`PsfImagingOptics`.

    .. [#yang2023aberration] Yang, X., Fu, Q., Elhoseiny, M., & Heidrich, W. (2023).
        Aberration-aware depth-from-focus. IEEE Transactions on Pattern Analysis and Machine Intelligence.
    .. [#li2021end] Li, Z., Hou, Q., Wang, Z., Tan, F., Liu, J., & Zhang, W. (2021).
        End-to-end learned single lens design using fast differentiable ray tracing.
        Optics Letters, 46(21), 5453-5456.
    .. [#chen2021optical] Chen, S., Feng, H., Pan, D., Xu, Z., Li, Q., & Chen, Y. (2021).
        Optical aberrations correction in postprocessing using imaging simulation.
        ACM Transactions on Graphics (TOG), 40(5), 1-15.
    """
    _inherent = system.PsfImagingOptics._inherent + ['surfaces']
    _external = system.PsfImagingOptics._external + [
        'imaging_model',
        'psf_type', 'psf_center',
        'coherent_tracing_samples', 'coherent_tracing_sampling_pattern',
        'fov_type', 'sampler', 'wl_reduction',
        'repetitions'
    ]

    def __init__(
        self,
        surfaces: surf.CoaxialSurfaceList,
        sensor: Sensor = None,
        imaging_model: ImagingModel = 'psf',
        perspective_focal_length: float = None,
        psf_type: PsfType = 'inc_rect',
        psf_center: PsfCenter = 'linear',
        fov_type: FovType = 'perspective',
        sampler: surf.Sampler = None,
        coherent_tracing_samples: int = 512,
        coherent_tracing_sampling_pattern: str = 'quadrapolar',
        wl_reduction: WlReduction = 'mean',
        pupil_type: PupilType = 'paraxial',
        repetitions: int = 1,
        **kwargs
    ):
        if imaging_model == 'backward':
            raise NotImplementedError()

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
        self.wl_reduction: WlReduction = wl_reduction  #: See :class:`CoaxialRayTracing`.
        self.pupil_type: PupilType = pupil_type  #: See :class:`CoaxialRayTracing`.
        self.repetitions: int = repetitions  #: See :class:`CoaxialRayTracing`.
        self.imaging_model: ImagingModel = imaging_model  #: See :class:`CoaxialRayTracing`.

        if self.sampler is None and len(self.surfaces) > 0:
            self.sampler = self.surfaces.first.aperture.sampler('random', 256)

    @utils.with_external
    def render_image_scene(self, scene: _sc.ImageScene, imaging_model: ImagingModel = 'psf', **kwargs) -> Ts:
        if imaging_model == 'psf':
            return system.PsfImagingOptics.render_image_scene(self, scene, **kwargs)
        elif imaging_model == 'forward_rt':
            return rto.ForwardRayTracingOptics.render_image_scene(self, scene, **kwargs)
        elif imaging_model == 'backward_rt':
            raise NotImplementedError()
        else:
            raise ValueError(f'Unknown imaging model: {imaging_model}')

    @utils.with_external
    def render_point_cloud_scene(self, scene: _sc.PointCloudScene, imaging_model: ImagingModel = 'psf', **kwargs) -> Ts:
        if imaging_model == 'psf':
            return system.PsfImagingOptics.render_point_cloud_scene(self, scene, **kwargs)
        elif imaging_model == 'forward_rt':
            return rto.ForwardRayTracingOptics.render_point_cloud_scene(self, scene, **kwargs)
        elif imaging_model == 'backward_rt':
            raise NotImplementedError()
        else:
            raise ValueError(f'Unknown imaging model: {imaging_model}')

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
        ref_idx = self.surfaces.last.material.n(out_ray.wl)
        out_ray = out_ray.march_to(self.surfaces.total_length, ref_idx)
        return out_ray

    @utils.with_external
    def trace_point(self, point: Ts, wl: Vector = None, sampler: surf.Sampler = None) -> BatchedRay:
        sampled = self.surfaces.first.sample(sampler)  # N_spp x 3
        d, _ = _make_direction(sampled, self.cam2lens(point).unsqueeze(-2))  # ... x N_spp|1 x 3
        ray = BatchedRay(sampled, d.unsqueeze(-3), wl.unsqueeze(-1))  # ... x N_wl x N_spp x 3
        out_ray = self.trace_ray(ray)  # ... x N_wl x N_spp
        return out_ray

    @torch.no_grad()
    def focus_to_(self, depth: Scalar) -> Self:
        """
        Make the system focus at ``depth`` by adjusting the distance between the last
        surface and image plane. The best distance is determined by minimizing mean squared radial
        distance of intersections of rays emitted from a point at ``depth`` and image plane.

        :param depth: Depth of focus.
        :type depth: float | Tensor
        :return: Self.
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
        ref_wl: Scalar = None,
        samples: int = 1024
    ) -> int:
        """
        .. warning::

            This method is experimental.
        """
        if ref_wl is None:
            ref_wl = base.Length.as_default(system.DEFAULT_WL, 'm')

        self._check_circ_aperture()
        self._check_circ_surf()

        depth = typing.scalar(depth, self.dtype, self.device)
        ref_wl = typing.scalar(ref_wl, self.dtype, self.device)

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

    @utils.with_external
    def entr_pupil(
        self,
        pupil_type: PupilType = 'paraxial',
        wl: Vector = None,
        wl_reduction: WlReduction = None,
        **kwargs
    ) -> PupilSpec:
        """
        Finds the entrance pupil of the system.

        :param str pupil_type: The method to determine the entrance pupil. See :class:`CoaxialRayTracing`.
        :param wl: Wavelengths to compute. Pupils depend on wavelength because refractive indices do.
        :type wl: float | Sequence[float] | Tensor
        :param str wl_reduction: The way to reduce wavelength dimension. See :class:`CoaxialRayTracing`.
        :return: Radius and z-coordinate in :ref:`LCS <guide_optics_rt_lcs>` of entrance pupil.
            A 2-tuple of 0D tensors.
        :rtype: tuple[Tensor, Tensor]
        """
        return self._pupil(True, pupil_type, wl, typing.cast(WlReduction, wl_reduction), **kwargs)

    @utils.with_external
    def exit_pupil(
        self,
        pupil_type: PupilType = 'paraxial',
        wl: Vector = None,
        wl_reduction: WlReduction = None,
        **kwargs
    ) -> PupilSpec:
        """
        Finds the exit pupil of the system.

        :param str pupil_type: The method to determine the exit pupil. See :class:`CoaxialRayTracing`.
        :param wl: Wavelengths to compute. Pupils depend on wavelength because refractive indices do.
        :type wl: float | Sequence[float] | Tensor
        :param str wl_reduction: The way to reduce wavelength dimension. See :class:`CoaxialRayTracing`.
        :return: Radius and z-coordinate in :ref:`LCS <guide_optics_rt_lcs>` of exit pupil.
            A 2-tuple of 0D tensors.
        :rtype: tuple[Tensor, Tensor]
        """
        return self._pupil(False, pupil_type, wl, typing.cast(WlReduction, wl_reduction), **kwargs)

    @utils.with_external
    def pupil_probe(self, entr: bool, ref_point: Ts, wl: Vector = None, samples: int = 512) -> PupilSpec:
        """
        Finds pupils by sampling points on the first surface sufficiently to cover its aperture, trace rays
        originated from an origin and passing through these points. Pupils are determined
        with range of valid rays.

        :param bool entr: Whether to find entrance pupil or exit pupil otherwise.
        :param Tensor ref_point: Coordinate of the origin from which rays originated
            in :ref:`LCS <guide_optics_rt_lcs>`. A tensor of shape ``(..., 3)``.
        :param wl: Wavelengths to compute. Pupils depend on wavelength because refractive indices do.
        :type wl: float | Sequence[float] | Tensor
        :param int samples: Number of samples on the first surface in vertical and horizontal directions.
        :return: Radius and z-coordinate in :ref:`LCS <guide_optics_rt_lcs>` of pupils.
            A 2-tuple of 0D tensors.
        :rtype: tuple[Tensor, Tensor]
        """
        # self._check_circ_aperture()
        # self._check_circ_surf()

        # ref_point = ref_point.unsqueeze(-2).unsqueeze(-3)  # ... x 1 x 1 x 3
        # points_on_s0 = self.surfaces.first.sample(sampler)  # points on surface 0, N_spp x 3
        # d_parallel, _ = _make_direction(points_on_s0, ref_point, True)  # ... x 1 x N_spp  x 3
        #
        # ray = BatchedRay(points_on_s0, d_parallel, wl.unsqueeze(-1), d_normed=True)  # ... x N_wl x N_spp
        # out_ray = self.surfaces(ray)
        #
        # valid = out_ray.valid.broadcast_to(out_ray.shape)  # ... x N_wl x N_spp
        # xy_valid = ray.o[..., :2].masked_fill(~valid.unsqueeze(-1), float('nan'))
        # xy_mean = xy_valid.nanmean(-2)  # ... x N_wl x 2
        # points_chief = torch.cat([xy_mean, torch.zeros_like(xy_mean[..., [0]])], -1)  # ... x N_wl x 1 x 3
        # d_chief, l0_chief = _make_direction(points_chief, ref_point)  # ... x 1 x 1( x 3)
        # chief_ray = BatchedRay(points_chief, d_chief, wl, 0.)  # ... x N_wl x 1
        raise NotImplementedError()

    @utils.with_external(exclude='sampler')
    def pupil_trace(self, entr: bool, wl: Vector = None, sampler: surf.Sampler = None) -> PupilSpec:
        """
        Finds pupils by tracing a bundle of rays emitted from a point located at the edge of stop.
        The focus of them is considered as a point on the edge of pupils.

        :param bool entr: Whether to find entrance pupil or exit pupil otherwise.
        :param wl: Wavelengths to compute. Pupils depend on wavelength because refractive indices do.
        :type wl: float | Sequence[float] | Tensor
        :param Callable sampler: See :class:`CoaxialRayTracing`.
        :return: Radius and z-coordinate in :ref:`LCS <guide_optics_rt_lcs>` of pupils.
            A 2-tuple of tensors of shape ``(N_wl,)``.
        :rtype: tuple[Tensor, Tensor]
        """
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
        """
        Finds pupils according to paraxial ray tracing.

        :param bool entr: Whether to find entrance pupil or exit pupil otherwise.
        :param wl: Wavelengths to compute. Pupils depend on wavelength because refractive indices do.
        :type wl: float | Sequence[float] | Tensor
        :return: Radius and z-coordinate in :ref:`LCS <guide_optics_rt_lcs>` of pupils.
            A 2-tuple of scalars (when the stop is pupil) or tensors of shape ``(N_wl,)``.
        :rtype: tuple[Tensor, Tensor]
        """
        point, sublist = self._pupil_prepare(entr)

        for s in sublist:
            if not isinstance(s, surf.ParaxialMixIn):
                raise RuntimeError(f'Paraxial behavior of surface {s.ctx.index} ({type(s).__name__}) is not defined')
            point = s.px_image_point(wl, point, not entr)  # (N_wl x )3
        r, z = point[..., 1], point[..., 2]  # N_wl or 0d
        return r, z

    @utils.with_external
    def wavefront_map(
        self,
        origin: Ts,
        wl: Vector = None,
        coherent_tracing_samples: int = DEFAULT_SAMPLES,
        coherent_tracing_sampling_pattern: str = 'quadrapolar',
    ) -> tuple[BatchedRay, Ts]:
        # This method is adapted from
        # https://github.com/TanGeeGo/ImagingSimulation/blob/master/PSF_generation/ray_tracing/difftrace/analysis.py
        chief_ray, ray, rs_roc, exit_pupil_distance = self._trace_opl_with_chief(
            origin, wl, coherent_tracing_samples, coherent_tracing_sampling_pattern
        )
        ref_idx = self.surfaces.mt_tail.n(ray.wl)
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
        Create a chief ray, i.e. one that passes through the center of entrance or exit pupil,
        originated from ``point``.

        :param Tensor point: Coordinate of the ray's origin in :ref:`LCS <guide_optics_rt_lcs>`.
            A tensor of shape ``(..., 3)``.
        :param wl: Wavelengths. Default: :attr:`.wl`.
        :type wl: float | Sequence[float] | Tensor
        :param str side: Which pupil (entrance or exit) to use, either ``'obj'``, ``'object'``,
            ``'img'`` or ``'image'``. Default: ``'obj'``.
        :param kwargs: Keyword arguments passed to :meth:`entr_pupil` or :meth:`exit_pupil`.
        :return: A chief ray with shape ``(..., N_wl)``.
        :rtype: BatchedRay
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
        d_chief, l0_chief = _make_direction(points_chief, origin, True)  # ... x 1 x 1( x 3)
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
        ref_idx = self.surfaces.mt_head.n(wl)
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
        ref_idx = self.surfaces.mt_tail.n(ray.wl)
        ray.march_(length2rs, ref_idx)
        return chief_ray, ray, rs_roc, exit_pupil_distance  # ... x N_wl x N_spp

    def _find_xy_center(self, psf_center, origins, out_ray, wl, wl_reduction: WlReduction):
        if psf_center == 'linear':
            xy_center = self.obj_proj_lens(origins)[..., None, None, :]  # ... x 1 x 1 x 2
        elif psf_center == 'mean':
            xy_center = out_ray.o[..., :2]  # ... x N_wl x N_spp x 2
            valid = out_ray.valid.unsqueeze(-1)  # ... x N_wl x N_spp x 1
            xy_center = torch.where(valid, xy_center, 0).sum(-2, True) / valid.sum(-2, True)  # ... x N_wl x 1 x 2
        elif psf_center == 'chief':
            origins = self.cam2lens(origins)
            chief = self.chief_ray(origins, wl, 'obj')  # ... x N_wl
            out_chief = self.trace_ray(chief)
            xy_center = out_chief.o[..., None, :2]  # ... x N_wl x 1 x 2
        else:
            raise ValueError(f'Unsupported PSF center type for simple incoherent PSF: {psf_center}')

        if psf_center != 'linear':
            if wl_reduction == 'none':
                pass  # xy_center: ... x N_wl x 1 x 2
            elif wl_reduction == 'mean':
                xy_center = xy_center.mean(-3, True)  # ... x 1 x 1 x 2
            elif wl_reduction == 'center':
                xy_center = xy_center[..., [xy_center.size(-3) // 2], :, :]  # ... x 1 x 1 x 2
            else:
                raise ValueError(f'Unknown WL reduction: {wl_reduction}')
        return xy_center

    @utils.with_external
    def _psf_inc_rect(
        self,
        origins: Ts,  # ... x 3
        psf_size: tuple[int, int],
        wl: Ts,  # N_wl
        psf_center: PsfCenter,
        sampler: surf.Sampler = None,
        wl_reduction: WlReduction = 'center',
    ) -> Ts:
        sampler = self.pick('sampler', sampler)
        f_name = self._psf_inc_rect.__name__

        out_ray = self.trace_point(origins, wl, sampler)  # ... x N_wl x N_spp
        out_ray.update_valid_(~out_ray.x.isnan() & ~out_ray.y.isnan())  # TODO:?
        out_ray = self.variable_hook(f'{f_name}.out_ray', out_ray)
        n_spp = out_ray.shape[-1]

        xy_center = self._find_xy_center(psf_center, origins, out_ray, wl, wl_reduction)
        xy = out_ray.o[..., :2]  # ... x N_wl x N_spp x 2
        # xy = torch.where(out_ray.valid.unsqueeze(-1), xy, 0)
        xy = xy - xy_center  # ... x N_wl x N_spp x 2

        xy = xy / self.new_tensor([self.sensor.pixel_size]).flip(0)
        xy = self.variable_hook(f'{f_name}.xy', xy)
        x, y = xy.unbind(-1)  # ... x N_wl x N_spp
        # if PSF size is odd, the center is N/2, relative positions are -N//2, ..., N//2
        # if PSF size is even, the center is (N+1)/2, relative positions are -N//2, ..., N//2-1
        x, y = x + (psf_size[1] // 2), y + (psf_size[0] // 2)
        c_a, r_a = torch.floor(x.detach() + 0.5).long(), torch.floor(y.detach() + 0.5).long()

        in_region = c_a.ge(0) & c_a.le(psf_size[1]) & r_a.ge(0) & r_a.le(psf_size[0])  # ... x N_wl x N_spp
        in_region = self.variable_hook(f'{f_name}.in_region', in_region)
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
        wl_reduction: WlReduction = 'center',
    ) -> Ts:
        sampler = self.pick('sampler', sampler)

        out_ray = self.trace_point(origins, wl, sampler)  # ... x N_wl x N_spp
        out_ray.update_valid_(~out_ray.x.isnan() & ~out_ray.y.isnan())  # TODO:?

        xy_center = self._find_xy_center(psf_center, self.cam2lens(origins), out_ray, wl, wl_reduction)
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

        ref_idx = self.surfaces.mt_tail.n(ray.wl)
        opd = chief_ray.march(-rs_roc, ref_idx).opl - ray.opl  # ... x N_wl x N_spp
        opd[~ray.valid] = float('nan')
        phase = opd * base.wave_vec(wl.unsqueeze(-1))

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

    def _pupil(
        self,
        entr: bool,
        pupil_type: PupilType,
        wl: Vector,
        wl_reduction: WlReduction,
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
