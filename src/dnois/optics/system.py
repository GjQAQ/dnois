import abc
import math
import random
import warnings

import torch

from . import formation, _func, psf_util
from .. import base, utils, scene as _sc, torch as _t
from ..base import ShapeError, PixelGrid, typing as ty

__all__ = [
    'BlendingModel',
    'ConvBlending',
    'DuplicatePsfOptics',
    'GeneralPsfRecenterType',
    'IdealOptics',
    'ImagingOptics',
    'PatchwiseConvBlending',
    'PinholeOptics',
    'PsfImagingOptics',
    'SegLit',
]

SegLit = ty.Literal['uniform', 'pointwise']
Seg = SegLit | ty.Double[int]
GeneralPsfRecenterType = bool | int | psf_util.PsfRecenterType

DEFAULT_WL = base.fline('d', 'He', unit='m')


class ImagingOptics(
    _t.TensorContainerMixIn,
    base.AsJsonMixIn,
    utils.ConvertSetAttrMixIn,
    torch.nn.Module,
):
    """
    Base class for all imaging optics.

    This class derives from :class:`torch.nn.Module`, whose input is
    a :class:`dnois.scene.Scene` object and output is rendered image.
    The rendered image may be single-channel, multi-channel,
    multi-wavelength or whatever, but must be a 2D array of pixels.
    The range of pixels are :math:`[0, 1]`.
    """

    def __init__(self, pixel_grid: PixelGrid = None):
        super().__init__()
        self.pixel_grid: PixelGrid | None = pixel_grid  #: Pixel grid.

    def render_image_scene(self, scene: _sc.ImageScene, **kwargs) -> ty.Ts:
        raise NotImplementedError(f'{self.render_image_scene.__qualname__} not implemented')

    def render_point_cloud_scene(self, scene: _sc.PointCloudScene, **kwargs) -> ty.Ts:
        raise NotImplementedError(f'{self.render_point_cloud_scene.__qualname__} not implemented')

    def render_view_array_scene(self, scene: _sc.ViewArrayScene, **kwargs) -> ty.Ts:
        raise NotImplementedError(f'{self.render_view_array_scene.__qualname__} not implemented')

    def forward(self, scene: _sc.Scene, **kwargs) -> ty.Ts:
        """
        Render a scene.

        :param dnois.scene.Scene scene: The scene to render.
        :param kwargs: Keyword arguments passed to ``self.render_*_scene`` methods.
        :return: Rendered image.
        :rtype: Tensor
        """
        if isinstance(scene, _sc.ImageScene):
            return self.render_image_scene(scene, **kwargs)
        elif isinstance(scene, _sc.PointCloudScene):
            return self.render_point_cloud_scene(scene, **kwargs)
        elif isinstance(scene, _sc.ViewArrayScene):
            return self.render_view_array_scene(scene, **kwargs)
        else:
            raise TypeError(f'Unknown scene type for {self._cn()}: {type(scene).__name__}')

    def pg(self) -> PixelGrid:
        """Get :attr:`.pixel_grid` but will raise an exception if it is ``None``."""
        pg = self.pixel_grid
        if pg is None:
            raise RuntimeError(f'An {self._cn()} object requires a pixel grid')
        return pg

    def _cn(self) -> str:  # just to avoid code to be too lengthy
        return self.__class__.__name__


class ObjectSpaceMixIn(_t.TensorContainerMixIn):  # TODO: check definition about angle unit
    def tanfovd2obj(self, tanfov: ty.Sequence[ty.Double[float]] | ty.Ts, depth: float | ty.Ts) -> ty.Ts:
        r"""
        Computes 3D coordinates of points in
        :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`
        given tangents of their FoV angles and depths:

        .. math::
            (x,y,z)=z(-\tan\varphi_x,-\tan\varphi_y,1)

        where :math:`z` indicates depth. Returned coordinates comply with
        :ref:`guide_imodel_ccs_inf`.

        :param tanfov: Tangents of FoV angles of points in radians. A tensor with shape ``(..., 2)``
            where the last dimension indicates x and y FoV angles. A list of 2-tuples of ``float``
            is seen as a tensor with shape ``(N, 2)``.
        :type tanfov: Sequence[tuple[float, float]] or Tensor
        :param depth: Depths of points. A tensor with any shape that is
            broadcastable with ``tanfov`` other than its last dimension.
        :type depth: float | Tensor
        :return: 3D coordinates of points, a tensor of shape ``(..., 3)``.
        :rtype: Tensor
        """
        if not torch.is_tensor(tanfov):
            tanfov = self.new_tensor(tanfov)
        if not torch.is_tensor(depth):
            depth = self.new_tensor(depth)
        _t.check_2d_vector(tanfov, f'tanfov in {self.tanfovd2obj.__qualname__}')

        tanx, tany, z = torch.broadcast_tensors(-tanfov[..., 0], -tanfov[..., 1], depth)  # ...
        is_inf = z.isinf()
        if is_inf.any():
            point_at_inf = torch.stack([tanx, tany, z], -1)  # ... x 3
            if is_inf.all():
                return point_at_inf
            else:
                return torch.where(is_inf, point_at_inf, torch.stack([tanx * z, tany * z, z], -1))
        else:
            return torch.stack([tanx * z, tany * z, z], -1)  # ... x 3

    def fovd2obj(
        self, fov: ty.Sequence[ty.Double[float]] | ty.Ts, depth: float | ty.Ts, in_degrees: bool = False
    ) -> ty.Ts:
        r"""
        Similar to :meth:`.tanfovd2obj`, but computes coordinates
        from FoV angles rather than their tangents.

        :param fov: FoV angles of points in radians. A tensor with shape ``(..., 2)``
            where the last dimension indicates x and y FoV angles.
        :type fov: Sequence[tuple[float, float]] or Tensor
        :param depth: Depths of points. A tensor with any shape that is
            broadcastable with ``fov`` other than its last dimension.
        :type depth: float | Tensor
        :param bool in_degrees: Whether ``fov`` is in degrees. If ``False``,
            ``fov`` is assumed to be in :doc:`default angle unit </content/guide/unit>`. Default: ``False``.
        :return: 3D coordinates of points, a tensor of shape ``(..., 3)``.
        :rtype: Tensor
        """
        if not torch.is_tensor(fov):
            fov = self.new_tensor(fov)
        if in_degrees:
            fov = fov.deg2rad()
        else:
            fov = base.Angle.default_to(fov, 'rad')
        if not torch.all(fov.gt(-torch.pi / 2) & fov.lt(torch.pi / 2)):
            raise ValueError(f'FoV angle must lie in the range (-pi/2, pi/2)')
        return self.tanfovd2obj(fov.tan(), depth)

    def obj2tanfov(self, point: ty.Ts) -> ty.Ts:
        r"""
        Converts coordinates of points in
        :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`
        into tangent of corresponding FoV angles:

        .. math::
            \tan\varphi_x=-x/z\\
            \tan\varphi_y=-y/z

        ``point`` complies with :ref:`guide_imodel_ccs_inf`.

        :param Tensor point: Coordinates of points. A tensor with shape ``(..., 3)``
            where the last dimension indicates coordinates of points in camera's coordinate system.
        :return: Tangent of x and y FoV angles. A tensor of shape ``(..., 2)``.
        :rtype: Tensor
        """
        _t.check_3d_vector(point, f'point in {self.obj2tanfov.__qualname__}')

        is_inf = point[..., [2]].isinf()  # ... x 1
        point2d = -point[..., :2]
        if is_inf.all():
            return point2d  # ... x 2
        else:
            if is_inf.any():
                return torch.where(is_inf, point2d, point2d / point[..., [2]])
            else:
                return point2d / point[..., [2]]

    def obj2fov(self, point: ty.Ts) -> ty.Ts:
        """
        Similar to :meth:`.point2tanfov`, but returns FoV angles rather than tangents.

        :param Tensor point: Coordinates of points. A tensor with shape ``(..., 3)``
            where the last dimension indicates coordinates of points in camera's coordinate system.
        :return: x and y FoV angles. A tensor of shape ``(..., 2)``.
        :rtype: Tensor
        """
        fov = self.obj2tanfov(point).arctan()
        fov = base.Angle.as_default(fov, 'rad')
        return fov


class PerspectiveMixIn(ObjectSpaceMixIn):
    @property
    @abc.abstractmethod
    def reference(self) -> 'PinholeOptics':
        """
        Returns the :ref:`reference model <guide_imodel_ref_model>` of  this object.

        :type: :class:`PinholeOptics`
        """
        pass

    def perspective(self, point: ty.Ts, flip: bool = True) -> ty.Ts:
        r"""
        Projects coordinates of points in
        :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`
        to image plane in a perspective manner:

        .. math::
            \left\{\begin{array}{l}
            x'=-\frac{f}{z}x
            y'=-\frac{f}{z}y
            \end{array}\right.

        where :math:`f` is the focal length of :ref:`reference model <guide_imodel_ref_model>`.
        The negative sign is eliminated if ``flip`` is ``True``.

        :param Tensor point: Coordinates of points. A tensor with shape ``(..., 3)``
            where the last dimension indicates coordinates of points in camera's coordinate system.
        :param bool flip: If ``True``, returns coordinates projected on flipped (virtual) image plane.
            Otherwise, returns those projected on original image plane.
        :return: Projected x and y coordinates of points. A tensor of shape ``(..., 2)``.
        :rtype: Tensor
        """
        xy = self.obj2tanfov(point) * self._perspective_focal_length()
        return -xy if flip else xy

    def points_grid(self, segments: ty.Size2d, depth: float | ty.Ts) -> ty.Ts:
        """
        Creates some points in object space, each of which is mapped to the center of
        one of non-overlapping patches on the image plane by perspective projection.

        :param segments: Number of patches in vertical (``N_y``) and horizontal (``N_x``) directions.
        :type segments: int or tuple[int, int]
        :param depth: Depth of resulted points. A ``float`` or a tensor of any shape ``(...)``.
        :type depth: float or Tensor
        :return: A tensor of shape ``(..., N_y, N_x, 3)`` representing the coordinates of points in
            :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`.
        :rtype: Tensor
        """
        segments = ty.size2d(segments)
        if not torch.is_tensor(depth):
            depth = self.new_tensor(depth)

        rm = self.reference
        tanfov_y, tanfov_x = base.grid(
            segments, (2 / segments[0], 2 / segments[1]),
            symmetric=True, device=self.device, dtype=self.dtype
        )
        tanfov_x, tanfov_y = tanfov_x * math.tan(rm.fov_half_x), tanfov_y * math.tan(rm.fov_half_y)
        tanfov_x, tanfov_y = torch.broadcast_tensors(tanfov_x, tanfov_y)  # N_x x N_y
        depth = depth[..., None, None]
        return self.tanfovd2obj(torch.stack([tanfov_x, tanfov_y], -1), depth)  # ... x N_x x N_y x 3

    def _perspective_focal_length(self) -> float:
        f = getattr(self, 'perspective_focal_length', None)
        if f is None:
            raise RuntimeError(f'A {self.__class__.__name__} requires perspective focal length')
        return f


class PinholeOptics(ImagingOptics, PerspectiveMixIn):
    def __init__(self, fl: float, pixel_grid: PixelGrid = None):
        super().__init__(pixel_grid)
        self.fl: float = fl

    def render_image_scene(self, scene: _sc.ImageScene, **kwargs) -> ty.Ts:
        """
        Render an image scene. Pinhole camera returns the :attr:`~dnois.scene.ImageScene.image`
        directly as long as its intrinsic (if exists) and resolution match those of the sensor.

        :param dnois.scene.ImageScene scene: The scene to render.
        :param kwargs: Not used.
        :return: Rendered image with shape identical to that of the image of ``scene``.
        :rtype: Tensor
        """
        if (si := scene.intrinsic) is not None:
            if not torch.allclose(si, self.intrinsic().broadcast_to(si.size(0), -1, -1)):
                raise NotImplementedError()
        pn = self.pixel_grid.n
        if (scene.height, scene.width) != pn:
            raise RuntimeError(f'Got {_sc.ImageScene.__name__} with shape {scene.height}x{scene.width}, '
                               f'but size of sensor is {pn[0]}x{pn[1]}')
        return scene.image

    def render_point_cloud_scene(self, scene: _sc.PointCloudScene, **kwargs) -> ty.Ts:
        raise NotImplementedError()

    def intrinsic(self, **kwargs) -> ty.Ts:
        """
        Intrinsic matrix of the pinhole camera.

        :param kwargs: Keyword arguments passed to :py:func:`torch.zeros`
            to construct the matrix.
        :return: A tensor of shape (3, 3).
        :rtype: Tensor
        """
        pg = self.pg()
        i = self.new_zeros((3, 3), **kwargs)
        i[0, 0] = self.fl / pg.spacing[1]
        i[1, 1] = self.fl / pg.spacing[0]
        i[2, 2] = 1
        i[0, 2] = pg.vol(1) / 2
        i[1, 2] = pg.vol(0) / 2
        return i

    @property
    def reference(self) -> 'PinholeOptics':
        return self

    @property
    def fov_full(self) -> float:
        r"""
        Full FoV of the pinhole camera:

        .. math::
            \varphi_\text{full}=2\arctan\frac{L}{2f}

        where :math:`L` is diagonal length and :math:`f` is focal length.

        :return: Full FoV in radian.
        :rtype: float
        """
        return self.fov_half * 2

    @property
    def fov_half(self) -> float:
        r"""
        Half FoV of the pinhole camera:

        .. math::
            \varphi_\text{half}=\arctan\frac{L}{2f}

        where :math:`L` is diagonal length and :math:`f` is focal length.

        :return: Half FoV in radian.
        :rtype: float
        """
        pg = self.pg()
        half_h, half_w = pg.vol(0) / 2, pg.vol(1) / 2
        tan = math.sqrt(half_h * half_h + half_w * half_w) / self.fl
        return math.atan(tan)

    @property
    def fov_half_x(self) -> float:
        r"""
        Half X-FoV of the pinhole camera:

        .. math::
            \varphi_\text{half-X}=\arctan\frac{W}{2f}

        where :math:`W` is the width of sensor and :math:`f` is focal length.

        :return: Half X-FoV in radian.
        :rtype: float
        """
        return math.atan(self.pg().vol(1) / (2 * self.fl))

    @property
    def fov_half_y(self) -> float:
        r"""
        Half Y-FoV of the pinhole camera:

        .. math::
            \varphi_\text{half-Y}=\arctan\frac{H}{2f}

        where :math:`H` is the height of sensor and :math:`f` is focal length.

        :return: Half Y-FoV in radian.
        :rtype: float
        """
        return math.atan(self.pg().vol(0) / (2 * self.fl))

    def _perspective_focal_length(self) -> float:
        return self.fl


def _symmetric_patch(obj_points: ty.Ts, x_symmetric: bool, y_symmetric: bool) -> ty.Ts:
    if x_symmetric:
        obj_points = obj_points[:, :(obj_points.size(1) + 1) // 2, :]
    if y_symmetric:
        obj_points = obj_points[:, :, :(obj_points.size(2) + 1) // 2]
    return obj_points


def _stitch_symmetric(psf: ty.Ts, h: int, w: int, x_symmetric: bool, y_symmetric: bool) -> ty.Ts:
    if x_symmetric:
        x_copy = psf[:, :h // 2, :].flip(1, -2)
        psf = torch.cat([psf, x_copy], 1)
    if y_symmetric:
        y_copy = psf[:, :, :w // 2].flip(2, -1)
        psf = torch.cat([psf, y_copy], 2)
    return psf


def _random_fov(lower: float, upper: float = None):
    if upper is None:
        lower, upper = -lower, lower
    lower, upper = math.tan(lower), math.tan(upper)
    value = random.uniform(lower, upper)
    value = math.atan(value)
    return value


# This class provides methods to inversely map points on image plane into object space.
class RenderImageSceneMixIn(PerspectiveMixIn, metaclass=abc.ABCMeta):
    depth: ty.Ts

    def random_fov(self):
        return _random_fov(self.fov_x_lower, self.fov_x_upper), _random_fov(self.fov_y_lower, self.fov_y_upper)

    @utils.with_external(exclude='depth')
    def points_grid(self, segments: ty.Size2d, depth: float | ty.Ts, depth_as_map: bool = False) -> ty.Ts:
        """
        Creates some points in object space, each of which is mapped to the center of
        one of non-overlapping patches on the image plane by perspective projection.

        :param segments: Number of patches in vertical (``N_y``) and horizontal (``N_x``) directions.
        :type segments: int or tuple[int, int]
        :param depth: Depth of resulted points. A ``float`` or a tensor of any shape ``(...)``.
            if ``depth_as_map`` is ``False``. Otherwise, must be a tensor of shape ``(..., N_y, N_x)``.
        :type depth: float or Tensor
        :param bool depth_as_map: See description of ``depth``.
        :return: A tensor of shape ``(..., N_y, N_x, 3)`` representing the coordinates of points in
            :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`.
        :rtype: Tensor
        """
        segments = ty.size2d(segments)
        if not torch.is_tensor(depth):
            depth = self.new_tensor(depth)
        if depth_as_map and depth.shape[-2:] != segments:
            raise ShapeError(f'The last two dimensions of depth map ({depth.shape[-2:]} '
                             f'must match the number of segments ({segments})')

        tanfov_y, tanfov_x = base.grid(
            segments, (1 / segments[0], 1 / segments[1]), 0.5,
            symmetric=True, device=self.device, dtype=self.dtype
        )
        tanfov_x = (1 - tanfov_x) * math.tan(self.fov_x_lower) + tanfov_x * math.tan(self.fov_x_upper)
        tanfov_y = (1 - tanfov_y) * math.tan(self.fov_y_lower) + tanfov_y * math.tan(self.fov_y_upper)
        tanfov_x, tanfov_y = torch.broadcast_tensors(tanfov_x, tanfov_y)  # N_x x N_y
        if not depth_as_map:
            depth = depth[..., None, None]
        return self.tanfovd2obj(torch.stack([tanfov_x, tanfov_y], -1), depth)  # ... x N_x x N_y x 3

    @property
    def fov_x_upper(self) -> float:
        """
        Maximum x FoV in radian.

        :type: float
        """
        return self.reference.fov_half_x

    @property
    def fov_x_lower(self) -> float:
        """
        Minimum x FoV in radian.

        :type: float
        """
        return -self.reference.fov_half_x

    @property
    def fov_x_full(self) -> float:
        """
        Full x FoV in radian. It is the difference between :attr:`.fov_x_upper` and :attr:`.fov_x_lower`.

        :type: float
        """
        return self.fov_x_upper - self.fov_x_lower

    @property
    def fov_y_upper(self) -> float:
        """
        Maximum y FoV in radian.

        :type: float
        """
        return self.reference.fov_half_y

    @property
    def fov_y_lower(self) -> float:
        """
        Minimum y FoV in radian.

        :type: float
        """
        return -self.reference.fov_half_y

    @property
    def fov_y_full(self) -> float:
        """
        Full y FoV in radian. It is the difference between :attr:`.fov_y_upper` and :attr:`.fov_y_lower`.

        :type: float
        """
        return self.fov_y_upper - self.fov_y_lower

    @staticmethod
    def seq_depth(
        depth: ty.Vector | ty.Double[ty.Ts] = None,
        sampling_curve: ty.Callable[[ty.Ts], ty.Ts] = None,
        n: int = None
    ) -> ty.Ts:
        r"""
        Returns a 1D tensor representing a series of depths, inferred from ``depth``:

        - If ``depth`` is a pair of 0D tensor, i.e. lower and upper bound of depth,
          returns a tensor with length ``n`` whose values are

          .. math::
            \text{depth}=\text{depth}_\min+(\text{depth}_\max-\text{depth}_\min)\times \Gamma(t).

          where :math:`t` is drawn uniformly from :math:`[0,1]`. An optional ``sampling_curve``
          (denoted by :math:`\Gamma`) can be given to control its values.
          By default, :math:`\Gamma` is constructed so that the inverse of depth is evenly spaced.
        - If a 1D tensor, returns it as-is.

        :param depth: See the eponymous argument of :class:`PsfImagingOptics` for details.
            Default: :attr:`.depth`.
        :param sampling_curve: Sampling curve :math:`\Gamma`,
            only makes sense in the first case above. Default: omitted.
        :type sampling_curve: Callable[[Tensor], Tensor]
        :param int n: Number of depths, only makes sense in the first case above. Default: omitted.
        :return: 1D tensor of depths.
        :rtype: Tensor
        """
        if isinstance(depth, tuple):  # [min, max]
            if n is None:
                raise TypeError(f'Number of depths must be given because only the lower and '
                                f'upper bound is specified for depth')
            d1, d2 = depth
            t = torch.linspace(0, 1, n, device=d2.device, dtype=d2.dtype)
            if sampling_curve is not None:
                t = sampling_curve(t)
            else:
                t = d1 * t / (d2 - (d2 - d1) * t)  # default sampling curve
            return d1 + (d2 - d1) * t
        elif sampling_curve is not None or n is not None:
            warnings.warn(f'sampling_curve and n are ignored because depth is already specified')
        return depth

    @staticmethod
    def random_depth(
        depth: ty.Vector | ty.Double[ty.Ts] = None,
        sampling_curve: ty.Callable[[ty.Ts], ty.Ts] = None,
        probabilities: ty.Ts = None
    ) -> ty.Ts:
        r"""
        Randomly sample a depth and returns it, inferred from ``depth``:

        - If ``depth`` is a pair of 0D tensor, i.e. lower and upper bound of depth, returns

          .. math::
            \text{depth}=\text{depth}_\min+(\text{depth}_\max-\text{depth}_\min)\times \Gamma(t).

          where :math:`t` is drawn uniformly from :math:`[0,1]`. An optional ``sampling_curve``
          (denoted by :math:`\Gamma`) can be given to control its distribution.
          By default, :math:`\Gamma` is constructed so that the inverse of depth is evenly spaced.
        - If a 1D tensor, randomly draws a value from it. Corresponding probability
          distribution can be given by ``probabilities``.

        :param depth: See the eponymous argument of :class:`PsfImagingOptics` for details.
            Default: :attr:`.depth`.
        :type depth: float, Sequence[float], Tensor or tuple[Tensor, Tensor]
        :param sampling_curve: Sampling curve :math:`\Gamma`,
            only makes sense in the first case above. Default: omitted.
        :type sampling_curve: Callable[[Tensor], Tensor]
        :param Tensor probabilities: A 1D tensor with same length as :attr:`.depth`,
            only makes sense in the third case above. Default: omitted.
        :return: A 0D tensor of randomly sampled depth.
        :rtype: Tensor
        """
        if isinstance(depth, tuple):  # randomly sampling from a depth range
            d1, d2 = depth
            t = torch.rand_like(d2)
            if sampling_curve is not None:
                t = sampling_curve(t)
            else:
                t = d1 * t / (d2 - (d2 - d1) * t)  # default sampling curve
            return d1 + (d2 - d1) * t
        else:  # randomly sampling from a set of depth values
            if probabilities is None:
                idx = torch.randint(depth.numel(), ()).item()
            else:
                if probabilities.ndim != 1:
                    raise ShapeError(f'probabilities must be a 1D tensor')
                idx = torch.multinomial(probabilities, 1).squeeze().item()
            return depth[idx]

    def _normalize_depth(self, depth: ty.Vector) -> ty.Ts:
        depth = ty.vector(depth, dtype=self.dtype, device=self.device)
        return depth

    def _make_depth_map(self, scene: _sc.ImageScene, depth: ty.Vector | ty.Double[ty.Ts]) -> ty.Ts:
        scene = scene.batch()
        n_b, _, n_h, n_w = scene.image.shape

        if scene.depth_aware:
            depth_map = scene.depth
        else:
            if not (torch.is_tensor(depth) and depth.numel() == 1):
                depth = torch.stack([self.random_depth(depth) for _ in range(n_b)])
            depth_map = depth.reshape(-1, 1, 1).expand(-1, n_h, n_w)
        return depth_map  # B|1 x H x W


class BlendingModel(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    A blending model indicates a method to render :ref:`imaged radiance field <guide_overview_irf>`
    given a clear image and its PSF. Objects of this class are used as part of
    :class:`PsfImagingOptics`. Although :class:`PsfImagingOptics` is responsible
    for the computation of PSF, relevant parameters are specified by this class
    so it calls :meth:`~PsfImagingOptics.psf` to obtain the PSF.
    """
    type: str  #: A class attribute to identify the type of the blending model.

    @abc.abstractmethod
    def forward(self, optics: 'PsfImagingOptics', scene: _sc.ImageScene, **kwargs) -> ty.Ts:
        pass

    @classmethod
    def create(cls, model_type: str, *args, **kwargs) -> ty.Self:
        """A factory method to instantiate a blending model given its type."""
        if cls is not BlendingModel:
            return cls(*args, **kwargs)  # noqa

        for sub in utils.subclasses(cls):
            if sub.type == model_type:
                return sub(*args, **kwargs)
        raise ValueError(f'Unknown blending model type: {model_type}')


class ConvBlending(BlendingModel):
    """
    Renders :ref:`imaged radiance field <guide_overview_irf>` via vanilla convolution.
    It means that PSF is considered as space-invariant.

    :param Module conv_model: A convolution model.
        Default: :class:`~dnois.optics.Simple`.
    """
    type = 'conv'

    def __init__(self, conv_model: torch.nn.Module = None):
        super().__init__()
        if conv_model is None:
            conv_model = formation.Simple()
        self.conv_model = conv_model

    def forward(
        self,
        optics: 'PsfImagingOptics',
        scene: _sc.ImageScene,
        fov: ty.Double[float] | ty.Callable[[], ty.Double[float]] | str = None,
        depth: ty.Vector = None,
        psf_cache: ty.Ts = None,
        **kwargs
    ) -> ty.Ts:
        r"""
        Forward method.

        :param optics: The imaging optics.
        :type optics: :class:`~dnois.optics.PsfImagingOptics`
        :param scene: The scene to be imaged.
        :type scene: :class:`~dnois.scene.ImageScene`
        :param fov: Corresponding FoV in degrees of the PSF used to render the scene.
            The argument is interpreted depending on its type:

            ``tuple[float, float]``
                x and y FoV angles.

            ``'random'``
                Randomly draw a pair of x and y FoV angles in a uniform distribution.

            ``Callable[[], tuple[float, float]]``
                A callable that returns a pair of x and y FoV angles.
                This is useful when non-uniform probability distribution is desired.

            Default: ``(0., 0.)``
        :param depth: A scalar representing object depth of the PSF.
            Default: ``optics.depth``. This argument is ignored if ``psf_cache`` is given.
        :type depth: float or 0D Tensor
        :param Tensor psf_cache: If given, use this tensor as PSF rather than compute it. Default: ``None``.
        :param kwargs: Additional keyword arguments passed to :meth:`PsfImagingOptics.psf`.
        :return: Computed :ref:`imaged radiance field <guide_overview_irf>`.
            A tensor of shape :math:`(B, N_\lambda, H, W)`.
        :rtype: Tensor
        """
        if fov is None:
            fov = (0., 0.)
        if isinstance(fov, str) and fov == 'random':
            fov = optics.random_fov()
        elif callable(fov):
            fov = ty.cast(ty.Callable, fov)()

        scene = scene.batch()  # (B,C,H,W)

        if psf_cache is None:
            if depth is None:
                depth = optics.depth.squeeze()
            depth = ty.scalar(depth)
            obj_points = optics.fovd2obj([fov], depth)  # (1,3)
            psf = optics.psf(obj_points, **kwargs)
        else:
            psf = psf_cache
        psf = optics.variable_hook(f'conv_render.psf', psf)

        # PSF: B(1) x N_wl x H_P x W_P
        image = self.conv_model(psf, scene.image)  # B x N_wl x H x W

        image = optics.crop(image)
        return image


class PatchwiseConvBlending(BlendingModel):
    """
    Renders :ref:`imaged radiance field <guide_overview_irf>` in a patch-wise manner.
    In other words, the image plane is partitioned into patches (overlapped or not)
    and PSF is assumed to be space-invariant in each patch, but varies
    from patch to patch.
    """
    type = 'patchwise'

    def forward(
        self,
        optics: 'PsfImagingOptics',
        scene: _sc.ImageScene,
        segments: ty.Size2d = (1, 1),
        depth: ty.Vector = None,
        pad: ty.Size2d = 0,
        linear_conv: bool = True,
        point_by_point: bool = False,
        psf_cache: ty.Ts = None,
        **kwargs
    ):
        r"""
        Forward method.

        :param PsfImagingOptics optics: The imaging optics.
        :param scene: The scene to be imaged.
        :type scene: :class:`~dnois.scene.ImageScene`
        :param segments: See :class:`PsfImagingOptics`. Default: ``(1, 1)``.
        :param depth: See :class:`PsfImagingOptics`. Default: ``optics.depth``.
        :param pad: Padding amount for each patch. See :func:`~dnois.optics.space_variant`
            for more details. Default: ``(0, 0)``.
        :type pad: int or tuple[int, int]
        :param bool linear_conv: Whether to compute linear convolution rather than
            circular convolution when computing blurred image. Default: ``True``.
        :param bool point_by_point: This method may take up huge amount of memory when
            ``segments`` is large. If ``point_by_point`` is ``True``, the method will
            compute PSFs of all patches one-by-one to ensure feasibility at the cost
            of computational efficiency. Default: ``False``.
        :param Tensor psf_cache: If given, use this tensor as PSF rather than compute it.
        :param kwargs: Additional keyword arguments passed to :meth:`PsfImagingOptics.psf`.
        :return: Computed :ref:`imaged radiance field <guide_overview_irf>`.
            A tensor of shape :math:`(B, N_\lambda, H, W)`.
        :rtype: Tensor
        """
        pad = ty.size2d(pad)

        if not isinstance(segments, tuple) or not len(segments) == 2 or not all(isinstance(s, int) for s in segments):
            raise ValueError(f'segments must be a pair of ints, got {segments}')

        scene = scene.batch()  # (B,C,H,W)

        if psf_cache is None:
            if depth is None:
                depth = optics.depth.squeeze()
            depth = ty.scalar(depth)
            obj_points = optics.points_grid(segments, depth)  # N_y x N_x x 3
            obj_points = _symmetric_patch(obj_points, optics.x_symmetric, optics.y_symmetric)
            if point_by_point:
                psf = torch.stack([
                    torch.stack([
                        optics.psf(p2, **kwargs) for p2 in p1.unbind()
                    ]) for p1 in obj_points.unbind()
                ])  # N_y x N_x x N_wl x H x W
            else:
                psf = optics.psf(obj_points, **kwargs)  # N_y x N_x x N_wl x H x W
            psf = _stitch_symmetric(psf, segments[0], segments[1], optics.x_symmetric, optics.y_symmetric)
        else:
            psf = psf_cache
        psf = optics.variable_hook('patchwise_render.psf', psf)

        psf = psf.permute(2, 0, 1, 3, 4)  # N_wl x N_y x N_x x H x W
        psf = psf.unsqueeze(0)
        image_blur = formation.space_variant(scene.image, psf, pad, linear_conv, point_by_point)  # B x N_wl x H x W

        image_blur = optics.crop(image_blur)
        return image_blur


class PsfImagingOptics(ImagingOptics, RenderImageSceneMixIn, utils.VarHookMixIn):
    """
    Base class for optical systems that renders images through PSFs.
    See :doc:`/content/guide/optics/imodel` for details.

    If two object points symmetric w.r.t. x-axis (i.e. whose x coordinates are equal
    and y coordinates are opposite) are expected to produce PSFs symmetric w.r.t. x-axis,
    one can set ``x_symmetric`` to ``True`` to compute PSFs in one side only,
    which is more efficient than computing two symmetric PSFs. It is similar
    for ``y_symmetric``. Specifically, an axisymmetric system allows both as ``True``.

    See :class:`ImagingOptics` for descriptions about more parameters.

    :param float perspective_focal_length: Focal length for perspective projection.
        Default: no perspective projection.
    :param wl: Wavelengths for imaging. Default: Fraunhofer *d* line.
        See :func:`~dnois.fraunhofer_line` for details.
    :type wl: float, Sequence[float] or 1D Tensor
    :param blending_model: Blending model used to render images.
        Default: :class:`ConvBlending`.
    :param depth: Depth adopted for rendering images when the scene to be imaged
        carries no depth information. Default: infinity.
    :type depth: float, Sequence[float] or Tensor
    :param psf_size: Height and width of PSF (i.e. convolution kernel) used to simulate imaging.
        Default: ``(64, 64)``.
    :type psf_size: int or tuple[int, int]
    :param bool norm_psf: Whether to normalize PSFs to have unit total energy. Default: ``True``.
    :param cropping: Widths in pixels for cropping after rendering to alleviate
        aliasing (caused by circular convolution) or dimming (caused by linear convolution)
        in edges. Default: 0.
    :param bool x_symmetric: Whether this system is symmetric w.r.t. x-axis.
        See descriptions above. Default: ``False``.
    :param bool y_symmetric: Whether this system is symmetric w.r.t. y-axis.
        See descriptions above. Default: ``False``.
    """
    wl: utils.Exparam
    blending_model: utils.Exparam
    depth: utils.Exparam
    psf_size: utils.Exparam
    norm_psf: utils.Exparam
    psf_recenter: utils.Exparam
    cropping: utils.Exparam
    x_symmetric: utils.Exparam
    y_symmetric: utils.Exparam

    def __init__(
        self,
        pixel_grid: PixelGrid = None,
        perspective_focal_length: float = None,
        wl: ty.Vector = None,
        blending_model: BlendingModel | str = 'conv',
        depth: ty.Vector = float('inf'),
        psf_size: ty.Size2d = 64,
        norm_psf: bool = True,
        psf_recenter: GeneralPsfRecenterType = False,
        cropping: ty.Size2d = 0,
        x_symmetric: bool = False,
        y_symmetric: bool = False,
    ):
        super().__init__(pixel_grid)
        if wl is None:
            wl = base.Length.as_default(DEFAULT_WL, 'm')  # self.wl is assumed to never be None

        self.register_buffer('depth', None)
        self.register_buffer('wl', None)

        self.perspective_focal_length: float | None = perspective_focal_length
        self.wl = wl  #: Wavelengths.
        self.depth = depth  #: Depths.
        #: Blending model.
        self.blending_model: BlendingModel = ty.cast(BlendingModel, blending_model)
        #: Height and width of PSF (i.e. convolution kernel) used to simulate imaging.
        #: See :class:`PsfImagingOptics`.
        self.psf_size: ty.Double[int] = ty.size2d(psf_size)
        self.norm_psf: bool = norm_psf  #: Whether to normalize PSFs to have unit total energy.
        self.psf_recenter: psf_util.PsfRecenter = ty.cast(psf_util.PsfRecenter, psf_recenter)
        self.cropping: ty.Double[int] = ty.size2d(cropping)  #: See :class:`PsfImagingOptics`.
        self.x_symmetric: bool = x_symmetric  #: See :class:`PsfImagingOptics`.
        self.y_symmetric: bool = y_symmetric  #: See :class:`PsfImagingOptics`.

    @abc.abstractmethod
    def psf(
        self,
        origins: ty.Ts = None,
        psf_size: ty.Size2d = None,
        wl: ty.Vector = None,
        norm_psf: bool = None,
        psf_recenter: GeneralPsfRecenterType = None,
        **kwargs
    ) -> ty.Ts:
        r"""
        Returns PSF of points whose coordinates
        in :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`
        are given by ``points``.

        The coordinate direction of returned PSF is defined as follows.
        Horizontal and vertical directions represent x- and y-axis, respectively.
        x is positive in left side and y is positive in upper side.
        In 3D space, the directions of x- and y-axis are identical to that of
        camera's coordinate system. In this way, returned PSF can be convolved with
        a clear image directly to produce a blurred image.

        :param Tensor origins: Source points of which to evaluate PSF. A tensor with shape
            ``(..., 3)`` where the last dimension indicates coordinates of points in camera's
            coordinate system. The coordinates comply with :ref:`guide_imodel_ccs_inf`.
            Default: the points corresponding to :attr:`.depth` and center FoV.
        :param psf_size: Numbers of pixels of PSF in vertical and horizontal directions.
            Default: :attr:`.psf_size`.
        :type psf_size: int or tuple[int, int]
        :param wl: Wavelengths to evaluate PSF on. Default: :attr:`.wl`.
        :type wl: float, Sequence[float] or Tensor
        :param bool norm_psf: Whether to normalize PSF to have unit total energy.
            Default: :attr:`.norm_psf`.
        :param psf_recenter:
            Default: :attr:`.psf_recenter`.
        :type psf_recenter: bool or int or str
        :return: PSF conditioned on ``origins``. A tensor with shape ``(..., N_wl, H, W)``.
        :rtype: Tensor
        """
        pass

    @utils.with_external
    def render_image_scene(self, scene: _sc.ImageScene, segments: SegLit | ty.Size2d = None, **kwargs) -> ty.Ts:
        r"""
        Implementation of :doc:`imaging simulation </content/guide/overview>`.
        This method will call either of three imaging methods:

        - If ``segments`` is ``'uniform'``, call :meth:`conv_render`;
        - If ``'pointwise'``, call :meth:`pointwise_render`;
        - Otherwise, ``segments`` is a pair of integers, call :meth:`patchwise_render`.

        :param scene: The scene to be imaged.
        :type scene: :class:`~dnois.scene.Scene`
        :param segments: See :class:`PsfImagingOptics`. Default: :attr:`.segments`.
        :param kwargs: Additional keyword arguments passed to the underlying imaging methods.
        :return: Computed :ref:`imaged radiance field <guide_overview_irf>`.
            A tensor of shape :math:`(B, N_\lambda, H, W)`.
        :rtype: Tensor
        """
        self._check_image_scene(scene)
        return self.blending_model(self, scene, **kwargs)

    def render_point_cloud_scene(self, scene: _sc.PointCloudScene, **kwargs) -> ty.Ts:
        raise NotImplementedError()

    @utils.with_external
    def psf_array(self, segments: ty.Size2d, depth: ty.Vector) -> ty.Ts:
        segments = ty.size2d(segments)
        obj_points = self.points_grid(segments, depth)  # (N_d,N_H,N_W,3)
        obj_points = _symmetric_patch(obj_points, self.x_symmetric, self.y_symmetric)
        psf = self.psf(obj_points)  # (N_d,N_H,N_W,N_wl,H,W)
        psf = _stitch_symmetric(psf, segments[0], segments[1], self.x_symmetric, self.y_symmetric)
        return psf

    def crop(self, image: ty.Ts) -> ty.Ts:
        """
        Crop ``image`` by width :attr:`.cropping`.

        :param Tensor image: A tensor of shape ``(..., H, W)``.
        :return: Cropped image. A tensor of shape ``(..., H', W')``.
        :rtype: Tensor
        """
        return utils.crop(image, self.cropping)

    @property
    def reference(self) -> 'PinholeOptics':
        return PinholeOptics(self._perspective_focal_length(), self.pg())

    def _check_image_scene(self, scene: _sc.Scene):
        if not isinstance(scene, _sc.ImageScene):
            raise RuntimeError(f'An {_sc.ImageScene.__name__} expected, but got {type(scene).__name__}')
        if scene.n_plr != 0:
            raise NotImplementedError(f'{self._cn()} does not support polarization currently')
        if scene.intrinsic is not None:
            raise NotImplementedError(f'{self._cn()} does not support scenes with intrinsic currently')

    # region External parameter normalizers

    def _normalize_wl(self, wl: ty.Vector):
        return ty.vector(wl, dtype=self.dtype, device=self.device)

    _normalize_psf_size = staticmethod(ty.size2d)
    _normalize_psf_recenter = staticmethod(utils.type_normalizer(psf_util.PsfRecenter))
    _normalize_blending_model = staticmethod(utils.type_normalizer(BlendingModel))

    # endregion

    # region Serialization

    def _todict_depth(self, keep_tensor: bool = True):
        depth = self.depth
        if keep_tensor:
            return depth
        if torch.is_tensor(depth):
            return depth.tolist()
        return {'min': depth[0].tolist(), 'max': depth[1].tolist()}

    def _todict_pixel_grid(self, keep_tensor: bool = True):
        if self.pixel_grid is None:
            return None
        else:
            return self.pixel_grid.to_dict(keep_tensor)

    @classmethod
    def _pre_from_dict(cls, d: dict):
        d = super()._pre_from_dict(d)
        depth = d['depth']
        if isinstance(depth, dict):
            d['depth'] = (torch.tensor(depth['min']), torch.tensor(depth['max']))
        if d['pixel_grid'] is not None:
            d['pixel_grid'] = PixelGrid.from_dict(d['pixel_grid'])
        return d

    # endregion


class DuplicatePsfOptics(PsfImagingOptics):
    """
    A simple model that duplicates the PSF of another :class:`PsfImagingOptics`.

    .. note::
        The coordinate system in following description is
        :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`.

    :param PsfImagingOptics source: The source :class:`PsfImagingOptics`.
    :param symmetry: Symmetry relationship between this optical system and the source.
        Either ``'x'`` (symmetric w.r.t. y-axis, the same below), ``'y'``,
        ``'central'`` (symmetric w.r.t. the origin), ``'diag'`` (symmetric w.r.t. the diagonal
        between x and y-axis), ``'adiag'`` (anti-diagonal) or ``None`` (identical).
        Default: ``None``.
    :type symmetry: str or None
    """
    source: PsfImagingOptics

    def __init__(
        self,
        source: PsfImagingOptics,
        symmetry: ty.Literal['x', 'y', 'central', 'diag', 'adiag'] = None,
        source_as_submodule: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        if source_as_submodule:
            self.source = source
        else:
            self.__dict__['source'] = source  # avoid submodule registration
        self.symmetry = symmetry

    def psf(self, origins: ty.Ts = None, *args, **kwargs) -> ty.Ts:
        if self.symmetry:
            origins = origins.clone()
        if self.symmetry == 'x':
            origins[..., 0] = -origins[..., 0]
        elif self.symmetry == 'y':
            origins[..., 1] = -origins[..., 1]
        elif self.symmetry == 'central':
            origins[..., :2] = -origins[..., :2]
        elif self.symmetry == 'diag':
            origins[..., [0, 1]] = origins[..., [1, 0]]
        elif self.symmetry == 'adiag':
            origins[..., [0, 1]] = -origins[..., [1, 0]]

        psf = self.source.psf(origins, *args, **kwargs)

        if self.symmetry == 'x':
            psf = torch.flip(psf, dims=(-1,))
        elif self.symmetry == 'y':
            psf = torch.flip(psf, dims=(-2,))
        elif self.symmetry == 'central':
            psf = torch.flip(psf, dims=(-1, -2))
        elif self.symmetry == 'diag':
            psf = psf.transpose(-2, -1)
        elif self.symmetry == 'adiag':
            psf = torch.flip(psf, dims=(-1, -2)).transpose(-2, -1)
        return psf


class IdealOptics(PsfImagingOptics):
    """
    Ideal optics model.

    See :class:`PsfImagingOptics` for descriptions of more parameters.

    :param float pupil_diameter: Diameter of the light-passing pupil on principal planes.
    :param float fl1: Focal length in object space.
    :param float fl2: Focal length in image space.
    :param kwargs: Additional keyword arguments passed to :class:`PsfImagingOptics`.
    """

    def __init__(
        self,
        pupil_diameter: float,
        fl1: float,
        fl2: float = None,
        pixel_grid: PixelGrid = None,
        perspective_focal_length: float = None,
        sensor_distance: float = None,
        **kwargs,
    ):
        if fl2 is None:
            fl2 = fl1
        if sensor_distance is not None:
            if perspective_focal_length is not None:
                raise ValueError(f'Only one of perspective_focal_length and sensor_distance can be specified')
            perspective_focal_length = sensor_distance * fl1 / fl2

        kwargs.setdefault('x_symmetric', True)
        kwargs.setdefault('y_symmetric', True)

        super().__init__(pixel_grid, perspective_focal_length, **kwargs)
        self.pupil_diameter: float = pupil_diameter  #: Diameter of the light-passing pupil on principal planes.
        self.fl1: float = fl1  #: Focal length in object space.
        self.fl2: float = fl2  #: Focal length in image space.

    @utils.with_external
    def psf(self, origins: ty.Ts = None, psf_size: ty.Size2d = None, **kwargs) -> ty.Ts:
        if len(kwargs) != 0:
            raise RuntimeError(f'Unknown keyword arguments for {self.__class__.__name__}: '
                               f'{", ".join(kwargs.keys())}')
        if origins is None:
            origins = self.tanfovd2obj([(0, 0)], self.depth)

        obj_d = origins[..., 2]  # ...
        img_d = _func.imgd(obj_d, self.fl1, self.fl2)  # ...
        coc = _func.circle_of_confusion(self.pupil_diameter, self.sensor_distance, img_d)  # ...
        radius = coc[..., None, None] / 2  # ... x 1 x 1

        psf = torch.zeros(*coc.shape, *psf_size, device=origins.device, dtype=origins.dtype)  # ... x H x W
        y, x = base.grid(psf_size, self.pg().spacing, device=origins.device, dtype=origins.dtype)
        r2 = x.square() + y.square()  # H x W
        psf[r2 <= radius.square()] = 1
        psf = psf_util.norm_psf(psf)
        psf = psf.unsqueeze(-3)  # ... x 1 x H x W
        return psf

    def patchwise_render(self, *args, **kwargs):
        warnings.warn(f'PSF of {IdealOptics} is always space-invariant so patch-wise rendering '
                      f'is virtually equivalent to vanilla convolution but far more inefficient. '
                      f'Consider using {self.conv_render.__name__} instead.')
        return super().patchwise_render(*args, **kwargs)

    def to_dict(self, keep_tensor=True) -> dict[str, ty.Any]:
        d = super().to_dict(keep_tensor)
        del d['x_symmetric'], d['y_symmetric']
        return d

    @property
    def sensor_distance(self) -> float:
        """
        Distance between image principal plane and image plane (sensor plane).

        :type: float
        """
        return self._perspective_focal_length() * self.fl2 / self.fl1

    @sensor_distance.setter
    def sensor_distance(self, value: float):
        self.perspective_focal_length = value * self.fl1 / self.fl2
