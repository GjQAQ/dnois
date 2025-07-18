import abc
import math
import random
import warnings

import torch

from . import formation, _func
from .. import base, utils, depth as _d, scene as _sc, torch as _t
from ..base import ShapeError, typing
from ..base.typing import (
    Ts, Size2d, Double, Vector, Callable,
    size2d, vector, cast
)
from ..sensor import Sensor

__all__ = [
    'IdealOptics',
    'ImagingOptics',
    'PinholeOptics',
    'PsfImagingOptics',
    'SegLit',
]

SegLit = typing.Literal['uniform', 'pointwise']
Seg = SegLit | Double[int]

DEFAULT_WL = base.fline('d', 'He', unit='m')


class ImagingOptics(
    _t.TensorContainerMixIn,
    base.AsJsonMixIn,
    utils.ExternalParamMixIn,
    torch.nn.Module,
    metaclass=abc.ABCMeta,
):
    """
    Base class for all imaging optics.

    :param Sensor sensor: The attached sensor. This can be left unspecified
        if no imaging simulation required.Default: no sensor.
    """

    def __init__(self, sensor: Sensor = None):
        super().__init__()
        self.sensor: Sensor | None = sensor  #: The attached sensor.

    def __setattr__(self, key, value):
        if key == 'sensor':
            self.__dict__[key] = value  # avoid it is registered as submodule
            return

        normalizer = getattr(self, '_normalize_' + key, None)
        if normalizer is not None:
            value = normalizer(value)
        return super().__setattr__(key, value)

    @abc.abstractmethod
    def render_image_scene(self, scene: _sc.ImageScene, **kwargs) -> Ts:
        pass

    @abc.abstractmethod
    def render_point_cloud_scene(self, scene: _sc.PointCloudScene, **kwargs) -> Ts:
        pass

    def forward(self, scene: _sc.Scene, **kwargs) -> Ts:
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
        else:
            raise TypeError(f'Unknown scene type for {self._cn()}: {type(scene).__name__}')

    def _cn(self) -> str:  # just to avoid code to be too lengthy
        return self.__class__.__name__

    def _sensor(self) -> Sensor:
        s = self.sensor
        if s is None:
            raise RuntimeError(f'An {self._cn()} object requires a sensor')
        return s


class ObjectSpaceMixIn(_t.TensorContainerMixIn):
    def tanfovd2obj(self, tanfov: typing.Sequence[Double[float]] | Ts, depth: float | Ts) -> Ts:
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
        self, fov: typing.Sequence[Double[float]] | Ts, depth: float | Ts, in_degrees: bool = False
    ) -> Ts:
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

    def obj2tanfov(self, point: Ts) -> Ts:
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

    def obj2fov(self, point: Ts) -> Ts:
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

    def perspective(self, point: Ts, flip: bool = True) -> Ts:
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

    def points_grid(self, segments: Size2d, depth: float | Ts) -> Ts:
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
        segments = size2d(segments)
        if not torch.is_tensor(depth):
            depth = self.new_tensor(depth)

        rm = self.reference
        tanfov_y, tanfov_x = utils.grid(
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
    def __init__(self, fl: float, sensor: Sensor = None):
        super().__init__(sensor)
        self.fl: float = fl

    def render_image_scene(self, scene: _sc.ImageScene, **kwargs) -> Ts:
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
        pn = self._sensor().pixel_num
        if (scene.height, scene.width) != pn:
            raise RuntimeError(f'Got {_sc.ImageScene.__name__} with shape {scene.height}x{scene.width}, '
                               f'but size of sensor is {pn[0]}x{pn[1]}')
        return scene.image

    def render_point_cloud_scene(self, scene: _sc.PointCloudScene, **kwargs) -> Ts:
        raise NotImplementedError()

    def intrinsic(self, **kwargs) -> Ts:
        """
        Intrinsic matrix of the pinhole camera.

        :param kwargs: Keyword arguments passed to :py:func:`torch.zeros`
            to construct the matrix.
        :return: A tensor of shape (3, 3).
        :rtype: Tensor
        """
        sensor = self._sensor()
        i = self.new_zeros((3, 3), **kwargs)
        i[0, 0] = self.fl / sensor.pixel_size[1]
        i[1, 1] = self.fl / sensor.pixel_size[0]
        i[2, 2] = 1
        i[0, 2] = sensor.w / 2
        i[1, 2] = sensor.h / 2
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
        half_h, half_w = self._sensor().h / 2, self._sensor().w / 2
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
        return math.atan(self._sensor().w / (2 * self.fl))

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
        return math.atan(self._sensor().h / (2 * self.fl))

    def _perspective_focal_length(self) -> float:
        return self.fl


def _symmetric_patch(obj_points: Ts, x_symmetric: bool, y_symmetric: bool) -> Ts:
    if x_symmetric:
        obj_points = obj_points[:, :(obj_points.size(1) + 1) // 2, :]
    if y_symmetric:
        obj_points = obj_points[:, :, :(obj_points.size(2) + 1) // 2]
    return obj_points


def _stitch_symmetric(psf: Ts, h: int, w: int, x_symmetric: bool, y_symmetric: bool) -> Ts:
    if x_symmetric:
        x_copy = psf[:, :h // 2, :].flip(1, -2)
        psf = torch.cat([psf, x_copy], 1)
    if y_symmetric:
        y_copy = psf[:, :, :w // 2].flip(2, -1)
        psf = torch.cat([psf, y_copy], 2)
    return psf


# This class provides methods to inversely map points on image plane into object space.
class RenderImageSceneMixIn(PerspectiveMixIn, metaclass=abc.ABCMeta):
    depth: Ts

    @utils.with_external
    def seq_depth(
        self,
        depth: Vector | Double[Ts] = None,
        sampling_curve: Callable[[Ts], Ts] = None,
        n: int = None
    ) -> Ts:
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

    @utils.with_external
    def random_depth(
        self,
        depth: Vector | Double[Ts] = None,
        sampling_curve: Callable[[Ts], Ts] = None,
        probabilities: Ts = None
    ) -> Ts:
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

    @utils.with_external(exclude='depth')
    def points_grid(self, segments: Size2d, depth: float | Ts, depth_as_map: bool = False) -> Ts:
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
        segments = size2d(segments)
        if not torch.is_tensor(depth):
            depth = self.new_tensor(depth)
        if depth_as_map and depth.shape[-2:] != segments:
            raise ShapeError(f'The last two dimensions of depth map ({depth.shape[-2:]} '
                             f'must match the number of segments ({segments})')

        tanfov_y, tanfov_x = utils.grid(
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

    def _normalize_depth(self, depth: Vector) -> Ts:
        depth = vector(depth, dtype=self.dtype, device=self.device)
        return depth

    def _make_depth_map(self, scene: _sc.ImageScene, depth: Vector | Double[Ts]) -> Ts:
        scene = scene.batch()
        n_b, _, n_h, n_w = scene.image.shape

        if scene.depth_aware:
            depth_map = scene.depth
        else:
            if not (torch.is_tensor(depth) and depth.numel() == 1):
                depth = torch.stack([self.random_depth(depth) for _ in range(n_b)])
            depth_map = depth.reshape(-1, 1, 1).expand(-1, n_h, n_w)
        return depth_map  # B|1 x H x W


def _random_fov(lower: float, upper: float = None):
    if upper is None:
        lower, upper = -lower, lower
    lower, upper = math.tan(lower), math.tan(upper)
    value = random.uniform(lower, upper)
    value = math.atan(value)
    return value


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
    :param segments: Number of field-of-view segments when rendering images.
        Default: ``'uniform'``.

        ``int`` or ``tuple[int, int]``
            The numbers of FoV segments in vertical and horizontal directions.
            PSFs in each segment are assumed to be FoV-invariant.

        ``'uniform'``
            PSF is assumed to be space-invariant hence simple convolution can be used.

        ``'pointwise'``
            The optical responses of every individual object points will be computed.
    :type segments: int, tuple[int, int] or str
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
    _inherent = ['sensor', 'perspective_focal_length']
    wl: utils.Exparam
    segments: utils.Exparam
    depth: utils.Exparam
    psf_size: utils.Exparam
    norm_psf: utils.Exparam
    cropping: utils.Exparam
    x_symmetric: utils.Exparam
    y_symmetric: utils.Exparam

    def __init__(
        self,
        sensor: Sensor = None,
        perspective_focal_length: float = None,
        wl: Vector = None,
        segments: SegLit | Size2d = 'uniform',
        depth: Vector = float('inf'),
        psf_size: Size2d = 64,
        norm_psf: bool = True,
        cropping: Size2d = 0,
        x_symmetric: bool = False,
        y_symmetric: bool = False,
    ):
        super().__init__(sensor)
        if wl is None:
            wl = base.Length.as_default(DEFAULT_WL, 'm')  # self.wl is assumed to never be None

        self.perspective_focal_length: float | None = perspective_focal_length
        self.wl = wl  # property setter
        self.depth = depth  # property setter
        #: Number of field-of-view segments when rendering images. See :class:`PsfImagingOptics`.
        self.segments: Seg = cast(Seg, segments)
        #: Height and width of PSF (i.e. convolution kernel) used to simulate imaging.
        #: See :class:`PsfImagingOptics`.
        self.psf_size: Double[int] = size2d(psf_size)
        self.norm_psf: bool = norm_psf  #: Whether to normalize PSFs to have unit total energy.
        self.cropping: Double[int] = size2d(cropping)  #: See :class:`PsfImagingOptics`.
        self.x_symmetric: bool = x_symmetric  #: See :class:`PsfImagingOptics`.
        self.y_symmetric: bool = y_symmetric  #: See :class:`PsfImagingOptics`.

    @abc.abstractmethod
    def psf(
        self,
        origins: Ts,
        psf_size: Size2d = None,
        wl: Vector = None,
        norm_psf: bool = None,
        **kwargs
    ) -> Ts:
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
        :param psf_size: Numbers of pixels of PSF in vertical and horizontal directions.
            Default: :attr:`.psf_size`.
        :type psf_size: int or tuple[int, int]
        :param wl: Wavelengths to evaluate PSF on. Default: :attr:`.wl`.
        :type wl: float, Sequence[float] or Tensor
        :param bool norm_psf: Whether to normalize PSF to have unit total energy.
            Default: :attr:`.norm_psf`.
        :return: PSF conditioned on ``origins``. A tensor with shape ``(..., N_wl, H, W)``.
        :rtype: Tensor
        """
        pass

    @utils.with_external
    def render_image_scene(self, scene: _sc.ImageScene, segments: SegLit | Size2d = None, **kwargs) -> Ts:
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
        if segments == 'uniform':
            return self.conv_render(scene, **kwargs)
        elif segments == 'pointwise':
            return self.pointwise_render(scene, **kwargs)
        else:  # tuple[int, int]
            return self.patchwise_render(scene, segments=segments, **kwargs)

    def render_point_cloud_scene(self, scene: _sc.PointCloudScene, **kwargs) -> Ts:
        raise NotImplementedError()

    @utils.with_external
    def pointwise_render(
        self,
        scene: _sc.ImageScene,
        wl: Vector = None,
        depth: Vector = None,
        psf_size: Size2d = None,
        norm_psf: bool = None,
        **kwargs,
    ) -> Ts:
        r"""
        Renders :ref:`imaged radiance field <guide_overview_irf>` in a point-wise manner,
        i.e. PSFs of all the pixels are computed and superposed.

        :param scene: The scene to be imaged.
        :type scene: :class:`~dnois.scene.Scene`
        :param wl: See :class:`PsfImagingOptics`. Default: :attr:`.wl`.
        :param depth: See :class:`PsfImagingOptics`. Default: :attr:`.depth`.
        :param psf_size: See :class:`PsfImagingOptics`. Default: :attr:`.psf_size`.
        :param bool norm_psf: See :class:`PsfImagingOptics`. Default: :attr:`.norm_psf`.
        :param kwargs: Additional keyword arguments passed to :meth:`.psf`.
        :return: Computed :ref:`imaged radiance field <guide_overview_irf>`.
            A tensor of shape :math:`(B, N_\lambda, H, W)`.
        :rtype: Tensor
        """
        self._check_image_scene(scene)
        if wl.numel() != scene.n_wl:
            raise ValueError(f'A scene with {wl.numel()} wavelengths expected, got {scene.n_wl}')

        scene = scene.batch()
        _, _, n_h, n_w = scene.image.shape
        depth_map = self._make_depth_map(scene, depth)  # B|1 x H x W
        obj_points = self.points_grid((n_h, n_w), depth_map, True)  # B|1 x H x W x 3
        if not scene.depth_aware:
            obj_points = _symmetric_patch(obj_points, self.x_symmetric, self.y_symmetric)

        psf = self.psf(obj_points, psf_size, wl, norm_psf, **kwargs)  # B|1 x H x W x N_wl x H_P x W_P
        if not scene.depth_aware:
            psf = _stitch_symmetric(psf, n_h, n_w, self.x_symmetric, self.y_symmetric)
        psf = psf.permute(0, 3, 1, 2, 4, 5)  # B|1 x N_wl x H x W x H_P x W_P

        image = formation.superpose(scene.image, psf)  # B x N_wl x H x W

        image = self.crop(image)
        return image

    @utils.with_external
    def patchwise_render(
        self,
        scene: _sc.ImageScene,
        pad: Size2d = 0,
        linear_conv: bool = True,
        segments: Size2d = None,
        wl: Vector = None,
        depth: Vector = None,
        psf_size: Size2d = None,
        norm_psf: bool = None,
        point_by_point: bool = False,
        **kwargs
    ) -> Ts:
        r"""
        Renders :ref:`imaged radiance field <guide_overview_irf>` in a patch-wise manner.
        In other words, the image plane is partitioned into non-overlapping
        patches and PSF is assumed to be space-invariant in each patch, but varies
        from patch to patch.

        :param scene: The scene to be imaged.
        :type scene: :class:`~dnois.scene.Scene`
        :param pad: Padding amount for each patch. See :func:`~dnois.optics.space_variant`
            for more details. Default: ``(0, 0)``.
        :type pad: int or tuple[int, int]
        :param bool linear_conv: Whether to compute linear convolution rather than
            circular convolution when computing blurred image. Default: ``True``.
        :param segments: See :class:`PsfImagingOptics`. Default: :attr:`.segments`.
        :param wl: See :class:`PsfImagingOptics`. Default: :attr:`.wl`.
        :param depth: See :class:`PsfImagingOptics`. Default: :attr:`.depth`.
        :param psf_size: See :class:`PsfImagingOptics`. Default: :attr:`.psf_size`.
        :param bool norm_psf: See :class:`PsfImagingOptics`. Default: :attr:`.norm_psf`.
        :param bool point_by_point: This method may take up huge amount of memory when
            ``segments`` is large. If ``point_by_point`` is ``True``, the method will
            compute PSFs of all patches one-by-one to ensure feasibility at the cost
            of computational efficiency. Default: ``False``.
        :param kwargs: Additional keyword arguments passed to :meth:`.psf`.
        :return: Computed :ref:`imaged radiance field <guide_overview_irf>`.
            A tensor of shape :math:`(B, N_\lambda, H, W)`.
        :rtype: Tensor
        """
        pad = size2d(pad)

        self._check_image_scene(scene)
        if not isinstance(segments, tuple) or not len(segments) == 2:
            raise ValueError(f'segments must be a pair of ints, got {type(segments)}')
        if wl.numel() != scene.n_wl:
            raise ValueError(f'A scene with {wl.numel()} wavelengths expected, got {scene.n_wl}')
        if scene.depth_aware:
            warnings.warn(f'Depth-aware rendering is not supported currently '
                          f'for {self.patchwise_render.__qualname__}')

        scene = scene.batch()
        if not depth.numel() == 1:
            depth = torch.stack([self.random_depth(depth) for _ in range(scene.image.size(0))])  # B(1)
        # B(1) x N_y x N_x x 3
        obj_points = self.points_grid(cast(Double[int], segments), depth.flatten())
        obj_points = _symmetric_patch(obj_points, self.x_symmetric, self.y_symmetric)

        psf_cache = kwargs.pop('_psf_cache', None)  # experimental feature
        if psf_cache is not None:
            psf = psf_cache
        else:
            if point_by_point:
                psf = torch.stack([
                    torch.stack([
                        torch.stack([
                            self.psf(p3, psf_size, wl, norm_psf, **kwargs) for p3 in p2.unbind()  # (3,)
                        ]) for p2 in p1.unbind()  # (N_x, 3)
                    ]) for p1 in obj_points.unbind()  # (N_y, N_x, 3)
                ])  # B(1) x N_y x N_x x N_wl x H x W
            else:
                psf = self.psf(obj_points, psf_size, wl, norm_psf, **kwargs)  # B(1) x N_y x N_x x N_wl x H x W
            psf = _stitch_symmetric(psf, segments[0], segments[1], self.x_symmetric, self.y_symmetric)
        psf = self.variable_hook('patchwise_render.psf', psf)

        psf = psf.permute(0, 3, 1, 2, 4, 5)  # B(1) x N_wl x N_y x N_x x H x W
        image_blur = formation.space_variant(scene.image, psf, pad, linear_conv, point_by_point)  # B x N_wl x H x W

        image_blur = self.crop(image_blur)
        return image_blur

    @utils.with_external
    def conv_render(
        self,
        scene: _sc.ImageScene,
        fov: Double[float] | Callable[[], Double[float]] | str = None,
        wl: Vector = None,
        depth: Vector = None,
        psf_size: Size2d = None,
        norm_psf: bool = None,
        pad: Size2d | str = 'linear',
        occlusion_aware: bool = False,
        depth_quantization_level: int = 16,
        compensate_edge: bool = False,
        eps: float = 1e-3,
        psf_cache: Ts = None,
        **kwargs
    ) -> Ts:
        r"""
        Renders :ref:`imaged radiance field <guide_overview_irf>` via vanilla convolution.
        It means that PSF is considered as space-invariant.

        :param scene: The scene to be imaged.
        :type scene: :class:`~dnois.scene.Scene`
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
        :param wl: See :class:`PsfImagingOptics`. Default: :attr:`.wl`.
        :param depth: See :class:`PsfImagingOptics`. Default: :attr:`.depth`.
        :param psf_size: See :class:`PsfImagingOptics`. Default: :attr:`.psf_size`.
        :param bool norm_psf: See :class:`PsfImagingOptics`. Default: :attr:`.norm_psf`.
        :param pad: Padding width used to mitigate aliasing. See :func:`dnois.fourier.dconv2`
            for more details. Default: ``'linear'``.
        :type pad: int, tuple[int, int] or str
        :param bool occlusion_aware: Whether to use occlusion-aware image formation algorithm.
            See :func:`dnois.optics.depth_aware` for more details.
            This matters only when ``scene`` carries depth map. Default: ``False``.
        :param int depth_quantization_level: Number of quantization levels for depth-aware imaging.
            This matters only when ``scene`` carries depth map. Default: ``16``.
        :param bool compensate_edge: See :func:`dnois.optics.simple`. Default: ``False``.
        :param float eps: See :func:`dnois.optics.simple`. Default: ``1e-3``.
        :param Tensor psf_cache: If given, use this tensor as PSF rather than compute it. Default: ``None``.
        :param kwargs: Additional keyword arguments passed to :meth:`.psf`.
        :return: Computed :ref:`imaged radiance field <guide_overview_irf>`.
            A tensor of shape :math:`(B, N_\lambda, H, W)`.
        :rtype: Tensor
        """
        if fov is None:
            fov = (0., 0.)
        if isinstance(fov, str) and fov == 'random':
            rm = self.reference
            fov = (_random_fov(rm.fov_half_x), _random_fov(rm.fov_half_y))
            fov = self.variable_hook('conv_render.fov', fov)
        elif callable(fov):
            fov = cast(Callable, fov)()

        self._check_image_scene(scene)
        if wl.numel() != scene.n_wl:
            raise ValueError(f'A scene with {wl.numel()} wavelengths expected, got {scene.n_wl}')

        scene = scene.batch()
        if scene.depth_aware:
            if not isinstance(depth, tuple):
                raise ValueError(f'depth must be a pair of 0D tensors for depth-aware imaging')
            q_depth = self.seq_depth(n=depth_quantization_level)  # D
            obj_points = self.fovd2obj([fov], q_depth)  # D x 3
        else:
            if not depth.numel() == 1:
                depth = torch.stack([self.random_depth(depth) for _ in range(scene.batch_size)])  # B(1)
            obj_points = self.fovd2obj([fov], depth)  # B(1) x 3

        if psf_cache is None:
            psf = self.psf(obj_points, psf_size, wl, norm_psf, **kwargs)
        else:
            psf = psf_cache
        psf = self.variable_hook(f'conv_render.psf', psf)

        if scene.depth_aware:
            psf = psf.transpose(0, 1)  # N_wl x D x H_P x W_P

            min_d, max_d = depth  # TODO: invalid branch
            masks = _d.quantize_depth_map(scene.depth, min_d, max_d, depth_quantization_level)  # D x B x H x W
            masks = masks.transpose(0, 1).unsqueeze(1)  # B x 1 x D x H x W
            image = formation.depth_aware(scene.image, masks, psf, pad, occlusion_aware)  # B x N_wl x H x W
        else:
            # PSF: B(1) x N_wl x H_P x W_P
            image = formation.simple(scene.image, psf, pad, compensate_edge, eps)  # B x N_wl x H x W

        image = self.crop(image)
        return image

    @utils.with_external
    def psf_array(self, segments: Size2d, depth: Vector) -> Ts:
        segments = size2d(segments)
        obj_points = self.points_grid(segments, depth)  # (N_d,N_H,N_W,3)
        obj_points = _symmetric_patch(obj_points, self.x_symmetric, self.y_symmetric)
        psf = self.psf(obj_points)  # (N_d,N_H,N_W,N_wl,H,W)
        psf = _stitch_symmetric(psf, segments[0], segments[1], self.x_symmetric, self.y_symmetric)
        return psf

    def crop(self, image: Ts) -> Ts:
        """
        Crop ``image`` by width :attr:`.cropping`.

        :param Tensor image: A tensor of shape ``(..., H, W)``.
        :return: Cropped image. A tensor of shape ``(..., H', W')``.
        :rtype: Tensor
        """
        return utils.crop(image, self.cropping)

    def to_dict(self, keep_tensor=True) -> dict[str, typing.Any]:
        d = {k: self._attr2dictitem(k, keep_tensor) for k in self._inherent}
        d.update({k: self._attr2dictitem(k, keep_tensor) for k in self._external})
        return d

    @property
    def reference(self) -> 'PinholeOptics':
        return PinholeOptics(self._perspective_focal_length(), self._sensor())

    # TODO: is it needed to be properties?
    @property
    def depth(self) -> Ts:
        """
        Depth values used when a scene has no depth information.
        A 1D Tensor. See :class:`PsfImagingOptics`.

        :type: Tensor
        """
        return self._b_depth

    @depth.setter
    def depth(self, value: Ts):  # already normalized in __setattr__
        self.register_buffer('_b_depth', value)

    @property
    def wl(self) -> Ts:
        """Wavelength for rendering. A 1D tensor.\n\n:type: Tensor"""
        return self._b_wl

    @wl.setter
    def wl(self, value: Ts):  # already normalized in __setattr__
        self.register_buffer('_b_wl', value)

    def _check_image_scene(self, scene: _sc.Scene):
        if not isinstance(scene, _sc.ImageScene):
            raise RuntimeError(f'An {_sc.ImageScene.__name__} expected, but got {type(scene).__name__}')
        if scene.n_plr != 0:
            raise NotImplementedError(f'{self._cn()} does not support polarization currently')
        if scene.intrinsic is not None:
            raise NotImplementedError(f'{self._cn()} does not support scenes with intrinsic currently')

    def _normalize_wl(self, wl: Vector):
        return vector(wl, dtype=self.dtype, device=self.device)

    @staticmethod
    def _normalize_segments(segments: SegLit | Size2d) -> Seg:
        if not isinstance(segments, str):
            segments = size2d(segments)
        return segments

    _normalize_psf_size = staticmethod(size2d)

    def _todict_depth(self, keep_tensor: bool = True):
        depth = self.depth
        if keep_tensor:
            return depth
        if torch.is_tensor(depth):
            return depth.tolist()
        return {'min': depth[0].tolist(), 'max': depth[1].tolist()}

    def _todict_sensor(self, _: bool = True):
        return None if self.sensor is None else {
            'pixel_size': self.sensor.pixel_size,
            'pixel_num': self.sensor.pixel_num,
        }

    @classmethod
    def _pre_from_dict(cls, d: dict):
        d = super()._pre_from_dict(d)
        depth = d['depth']
        if isinstance(depth, dict):
            d['depth'] = (torch.tensor(depth['min']), torch.tensor(depth['max']))
        if d['sensor'] is not None:
            d['sensor'] = Sensor(**d['sensor'])
        return d


class IdealOptics(PsfImagingOptics):
    """
    Ideal optics model.

    See :class:`PsfImagingOptics` for descriptions of more parameters.

    :param float pupil_diameter: Diameter of the light-passing pupil on principal planes.
    :param float fl1: Focal length in object space.
    :param float fl2: Focal length in image space.
    :param kwargs: Additional keyword arguments passed to :class:`PsfImagingOptics`.
    """
    _inherent = PsfImagingOptics._inherent + ['pupil_diameter', 'fl1', 'fl2']

    def __init__(
        self,
        pupil_diameter: float,
        fl1: float,
        fl2: float = None,
        sensor: Sensor = None,
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

        super().__init__(sensor, perspective_focal_length, **kwargs)
        self.pupil_diameter: float = pupil_diameter  #: Diameter of the light-passing pupil on principal planes.
        self.fl1: float = fl1  #: Focal length in object space.
        self.fl2: float = fl2  #: Focal length in image space.

    @utils.with_external
    def psf(self, origins: Ts, psf_size: Size2d = None, **kwargs) -> Ts:
        if len(kwargs) != 0:
            raise RuntimeError(f'Unknown keyword arguments for {self.__class__.__name__}: '
                               f'{", ".join(kwargs.keys())}')

        obj_d = origins[..., 2]  # ...
        img_d = _func.imgd(obj_d, self.fl1, self.fl2)  # ...
        coc = _func.circle_of_confusion(self.pupil_diameter, self.sensor_distance, img_d)  # ...
        radius = coc[..., None, None] / 2  # ... x 1 x 1

        psf = torch.zeros(*coc.shape, *psf_size, device=origins.device, dtype=origins.dtype)  # ... x H x W
        y, x = utils.grid(psf_size, self.sensor.pixel_size, device=origins.device, dtype=origins.dtype)
        r2 = x.square() + y.square()  # H x W
        psf[r2 <= radius.square()] = 1
        psf = _func.norm_psf(psf)
        psf = psf.unsqueeze(-3)  # ... x 1 x H x W
        return psf

    def pointwise_render(self, *args, **kwargs):
        warnings.warn(f'PSF of {IdealOptics} is always space-invariant so point-wise rendering '
                      f'is virtually equivalent to vanilla convolution but far more inefficient. '
                      f'Consider using {self.conv_render.__name__} instead.')
        return super().pointwise_render(*args, **kwargs)

    def patchwise_render(self, *args, **kwargs):
        warnings.warn(f'PSF of {IdealOptics} is always space-invariant so patch-wise rendering '
                      f'is virtually equivalent to vanilla convolution but far more inefficient. '
                      f'Consider using {self.conv_render.__name__} instead.')
        return super().patchwise_render(*args, **kwargs)

    def to_dict(self, keep_tensor=True) -> dict[str, typing.Any]:
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
