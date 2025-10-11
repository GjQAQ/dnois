from dataclasses import dataclass, field
import math
import warnings

import torch

from .. import surf
from ..ray import BatchedRay
from .... import base, ext, utils

__all__ = [
    'draw_rays',
    'draw_surfaces',
    'draw_surface_circular_stop',
    'draw_surface_fresnel',
    'draw_surface_thin_lens',
    'draw_surf_common',

    'CRTSpotDiagram',
    'CRTVisConfig',
]

ty = base.typing
if ty.TYPE_CHECKING:
    if ext.vis.mpl_available():
        from matplotlib.axes import Axes
        from matplotlib.pyplot import Figure
    else:
        Axes = ...
        Figure = ...


def _fov_linestyle(n: int) -> list[str]:
    bases = ['-', '--', '-.', ':']
    segs, rem = divmod(n, len(bases))
    if segs > 0:
        warnings.warn('More than 4 linestyles are being used. Some linestyles may be repeated')
    lss = bases * segs + bases[:rem]
    return lss


def _plot_rays_3d(ax, ray1: BatchedRay, ray2: BatchedRay, colors: list[str], lss: list[str]):
    # shape: N_fov x N_wl x N_spp x 3
    for ls, fov_slc1, fov_slc2, v in zip(lss, ray1.o, ray2.o, ray1.valid):  # N_wl x N_spp x 3
        for clr, wl_slc1, wl_slc2, vv in zip(colors, fov_slc1, fov_slc2, v):  # N_spp x 3
            ax.plot(
                (utils.t4plot(wl_slc1[:, 2][vv]), utils.t4plot(wl_slc2[:, 2][vv])),
                (utils.t4plot(wl_slc1[:, 1][vv]), utils.t4plot(wl_slc2[:, 1][vv])),
                color=clr, linestyle=ls, linewidth=0.75
            )


@dataclass
class CRTVisConfig:
    """A data class describing visualization configuration for :class:`CoaxialRayTracing`."""
    color_fresnel: str = 'orange'  #: Color of latent profile of Fresnel surface.
    #: Line style to draw profiles of surfaces.
    linestyle_surface: dict = field(default_factory=lambda: {'color': 'black', 'linewidth': 1})
    #: Line sytle to draw plane of object or image.
    linestyle_terminal: dict = field(default_factory=lambda: {'color': 'black', 'linewidth': 2})
    surface_points: int = 100  #: Number of points to draw profiles of surfaces.


@dataclass
class CRTSpotDiagram:
    """A data class encapsulating the information about a spot diagram of :class:`CoaxialRayTracing`."""
    fig: 'Figure'
    rms: ty.Ts = None
    geo_radius: ty.Ts = None


def draw_rays(
    ax: 'Axes',
    sq: surf.CoaxialSurfaceSequence,
    ray: BatchedRay,
    isinf: bool,
    height: ty.Ts,
    wl: ty.Ts,
    legend: bool,
):
    # ray: N_fov x N_wl x N_spp
    colors = [utils.wl2rgb(_wl, output_format='hex') for _wl in wl.tolist()]
    lss = _fov_linestyle(height.numel())

    if isinf:
        ray.broadcast_().march_to_(ray.new_tensor(0.))

    rays = [ray.broadcast_()]
    intercepted_rays = []
    handles = []
    for s in sq:
        handles.append(s.register_variable_hook('forward.intercepted', intercepted_rays.append))
        handles.append(s.register_variable_hook('forward.interacted', rays.append))
    intercepted_rays.append(sq.trace_out(ray).broadcast_())
    for h in handles:
        h.remove()

    for i in reversed(list(range(len(rays) - 1))):
        rays[i].valid = sq[i].backward_valid(rays[i + 1].valid)
    for ray1, ray2 in zip(rays, intercepted_rays):
        _plot_rays_3d(ax, ray1, ray2, colors, lss)

    if legend:
        import matplotlib.lines
        color_lines = [matplotlib.lines.Line2D([], [], color=c, linewidth=0.75) for c in colors]
        color_labels = [base.Length.fmt(_wl, 'um') for _wl in wl.tolist()]
        fov_lines = [matplotlib.lines.Line2D([], [], color='black', linestyle=ls, linewidth=0.75) for ls in lss]
        if isinf:
            fov_labels = [fr'${utils.fmt(math.degrees(math.atan(-h)))}^\circ$' for h in height.tolist()]
        else:
            fov_labels = [fr'${base.Length.fmt(h)}$' for h in height.tolist()]
        ax.legend(color_lines + fov_lines, color_labels + fov_labels)


def draw_surface_circular_stop(ax: 'Axes', sf: surf.CircularStop, config: CRTVisConfig):
    r = sf.apt.radius.item()
    length = r / 5
    z = sf.ctx.baseline.item()
    ax.plot(
        [[z, z, z - length / 2, z - length / 2], [z, z, z + length / 2, z + length / 2]],
        [[r, -r, r, -r], [r + length, -r - length, r, -r]],
        **config.linestyle_surface
    )
    return None


def draw_surface_thin_lens(ax: 'Axes', sf: surf.ThinLens, config: CRTVisConfig):
    z = draw_surf_common(ax, sf, config)

    radius = sf.apt.radius.item()
    length = radius / (10 * 2 ** 0.5)
    z0 = sf.ctx.baseline.item()
    ax.plot(
        [[z0, z0, z0, z0], [z0 - length, z0 + length, z0 + length, z0 - length]],
        [
            [radius, radius, -radius, -radius],
            [radius - length, radius - length, length - radius, length - radius]
        ],
        **config.linestyle_surface
    )
    return z


def draw_surface_fresnel(ax: 'Axes', sf: surf.Fresnel, config: CRTVisConfig):
    radius = sf.apt.radius.item()
    y = torch.linspace(-radius, radius, config.surface_points, device=sf.device)
    z_latent = sf.h(..., ..., y.square(), sf.virtual_wrapping) + sf.ctx.baseline

    latent_ls = config.linestyle_surface.copy()
    latent_ls['color'] = config.color_fresnel
    ax.plot(utils.t4plot(z_latent), utils.t4plot(y), **latent_ls)
    return z_latent[-1].item()


def draw_surf_common(ax: 'Axes', sf: surf.Surface, config: CRTVisConfig):
    radius = sf.apt.max_radius().item()
    y = torch.linspace(-radius, radius, config.surface_points, device=sf.device)
    z = sf.h(torch.zeros_like(y), y) + sf.ctx.baseline
    ax.plot(utils.t4plot(z), utils.t4plot(y), **config.linestyle_surface)
    return z[-1].item()


surface_drawer: dict[type[surf.Surface], callable] = {
    surf.CircularStop: draw_surface_circular_stop,
    surf.Fresnel: draw_surface_fresnel,
    surf.ThinLens: draw_surface_thin_lens,
}


def draw_surfaces(
    ax: 'Axes', surfaces: surf.CoaxialSurfaceSequence, config: CRTVisConfig
):  # TODO: handle infinite radius
    edge_z = []
    for sf in surfaces:  # surfaces
        if sf.__class__ in surface_drawer:
            z = surface_drawer[sf.__class__](ax, sf, config)
        else:
            z = draw_surf_common(ax, sf, config)
        edge_z.append(z)

    for i in range(len(surfaces) - 1):  # edges
        if surfaces[i].material.name in {'vacuum', 'air'}:
            continue

        r1 = surfaces[i].apt.radius.item()
        r2 = surfaces[i + 1].apt.radius.item()
        r = max(r1, r2)
        ax.plot(
            [[edge_z[i], edge_z[i]], [edge_z[i + 1], edge_z[i + 1]]],
            [[r, -r], [r, -r]],
            **config.linestyle_surface
        )
        if r1 != r2:
            if r1 > r2:
                z = edge_z[i + 1]
            else:
                z = edge_z[i]
                r1, r2 = r2, r1
            ax.plot([[z, z], [z, z]], [[r2, -r2], [r1, -r1]], **config.linestyle_surface)
