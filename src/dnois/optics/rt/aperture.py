import abc
import functools

import torch
from torch import nn

from .ray import BatchedRay
from ... import base, conf, torch as _t, utils
from ...base import typing as ty
from ...base.typing import Ts

__all__ = [
    'AnnularAperture',
    'Aperture',
    'BoundedAperture',
    'CircularAperture',
    'DummyAperture',
    'Sampler',
]

Sampler = ty.Callable[[], tuple[Ts, Ts]]


def _default_aperture_size():
    return base.Length.as_default(conf.default_aperture_radius, 'm')


class Aperture(_t.EnhancedModule, metaclass=abc.ABCMeta):
    """
    Base class for aperture shapes. Aperture refers to the region on a surface
    where rays can transmit. The region outside the aperture is assumed to be
    completely opaque. Note that the region inside is not necessarily completely
    transparent. The most common aperture type is :class:`CircularAperture`.

    Mathematically, an aperture is defined by a set of 2D points on the baseline
    plane of associated surface. See :class:`Surface` for more details.
    """

    @abc.abstractmethod
    def evaluate(self, x: Ts, y: Ts) -> torch.BoolTensor:
        """
        Returns a boolean tensor representing whether each point :math:`(x,y)` is
        inside the aperture. To jointly represent 2D coordinates, ``x`` and ``y``
        must be broadcastable.

        :param Tensor x: x coordinates of the points.
        :param Tensor y: y coordinates of the points.
        :return: See description above. The shape of returned tensor is the
            broadcast result of ``x`` and ``y``.
        :rtype: Tensor
        """
        pass

    def forward(self, ray: BatchedRay) -> BatchedRay:
        """
        Similar to :meth:`.evaluate`, but operates on rays.

        :param BatchedRay ray: Incident rays.
        :return: New rays among which those outside the aperture are marked as invalid.
        :rtype: BatchedRay
        """
        return ray.update_valid(self.pass_ray(ray))

    def pass_ray(self, ray: BatchedRay) -> torch.BoolTensor:
        """
        Similar to :meth:`.evaluate`, but operates on rays.

        :param BatchedRay ray: Incident rays.
        :return: A mask tensor indicating whether corresponding rays can pass the aperture.
        :rtype: torch.BoolTensor
        """
        return self.evaluate(ray.x, ray.y)

    def sample(self, mode: str, *args, **kwargs) -> tuple[Ts, Ts]:
        """
        Samples points on this aperture, i.e. baseline plane of associated surface.
        Specific distribution depends on ``mode``.

        :param str mode: Sampling mode. Calling object of this method should possess a
            ``sample_{mode}`` method.
        :return: Two 1D tensors of representing x and y coordinates of the points.
        :rtype: tuple[Tensor, Tensor]
        """
        meth_name = 'sample_' + mode
        meth: ty.Callable = getattr(self, meth_name, ...)
        if meth is ...:
            raise ValueError(f'Unknown sampling mode for {self.__class__.__name__}: {mode}')
        return meth(*args, **kwargs)

    def sample_center(self) -> tuple[Ts, Ts]:
        """
        Return the central point of the aperture.

        :return: Two ``[0.]`` tensors.
        :rtype: tuple[Tensor, Tensor]
        """
        return self.new_tensor([0.]), self.new_tensor([0.])

    def sampler(self, mode: str, *args, **kwargs) -> Sampler:
        """
        Returns a callable object that can be used to sample points on this aperture.
        When it is called, it will call :meth:`sample` with given arguments and return its return value.
        See :meth:`sample` for more details.

        :return: A callable object that can be used to sample points on this aperture.
        :rtype: Callable
        """
        return functools.partial(self.sample, mode, *args, **kwargs)

    def to_dict(self, keep_tensor=True) -> dict[str, ty.Any]:
        return {'type': self.__class__.__name__}

    @classmethod
    def from_dict(cls, d: dict):
        if cls is not Aperture:
            d.pop('type')
            return cls(**d)  # default implementation of eponymous method

        _ty = d['type']
        subs = utils.subclasses(cls)
        for sub in subs:
            if sub.__name__ == _ty:
                return ty.cast(type[Aperture], sub).from_dict(d)  # Calling eponymous method of subclass
        aperture_types = [sub.__name__ for sub in subs]
        raise RuntimeError(utils.invalid_option_msg('aperture type', _ty, aperture_types))


class DummyAperture(Aperture):
    """A dummy aperture that never blocks any ray."""

    def evaluate(self, x: Ts, y: Ts) -> torch.BoolTensor:
        shape = torch.broadcast_shapes(x.shape, y.shape)
        return ty.cast(torch.BoolTensor, self.new_ones(shape, dtype=torch.bool))


class BoundedAperture(Aperture, metaclass=abc.ABCMeta):
    """
    A base class for bounded apertures, but the bounds may be infinite here.

    For various sampling methods, the points outside the aperture will be discarded.
    So the number of points they returned may be less than the requested number.
    """

    @abc.abstractmethod
    def max_radius(self) -> Ts:
        """
        :return: Radius of a max circle centered at origin to cover the aperture.
        :rtype: 0D tensor"""
        pass

    @abc.abstractmethod
    def max_width(self) -> Ts:
        """
        :return: Width of a maximum rectangle centered at origin to cover the aperture.
        :rtype: 0D tensor"""
        pass

    def max_height(self) -> Ts:
        """
        :return: Height of a maximum rectangle centered at origin to cover the aperture.
        :rtype: 0D tensor"""
        return self.max_width()

    def sample_random(self, n: int) -> tuple[Ts, Ts]:
        """
        Returns at most ``n`` points randomly sampled on this aperture.

        :param int n: Maximum number of points.
        :return: Two 1D tensors of length less than ``n``, representing
            x and y coordinates of the points.
        :rtype: tuple[Tensor, Tensor]
        """
        x, y = self._sample_random(n)
        return self._discard_invalid(x, y)

    def sample_rect(self, n: ty.Size2d) -> tuple[Ts, Ts]:
        r"""
        Samples points on this aperture in an evenly spaced rectangular grid,
        where number of points in vertical and horizontal directions :math:`(H, W)`
        are given by ``n``. Note that the points outside the aperture are dropped
        so total number of returned points is less than :math:`HW`.

        :param n: A pair of int representing :math:`(H, W)`.
        :type n: int | tuple[int, int]
        :return: Two 1D tensors of representing x and y coordinates of the points.
        :rtype: tuple[Tensor, Tensor]
        """
        x, y = self._sample_rect(n)
        return self._discard_invalid(x, y)

    def sample_unipolar(self, n_radius: int = 6, n_angle: int = 6) -> tuple[Ts, Ts]:
        r"""
        Samples points on this aperture in a unipolar manner. Specifically, the aperture
        is divided into :math:`N_r` rings with equal widths and points are sampled on the
        outer edge of each ring. The first ring contains :math:`N_\theta` points, the second
        contains :math:`2N_\theta` points ... and so on, plus a point at center.
        Thus, there are totally :math:`N_\theta N_r(N_r+1)/2+1` points at most.

        :param int n_radius: Number of rings :math:`N_r`. Default: 6.
        :param int n_angle: Level of points increase per ring :math:`N_\theta`. Default: 6.
        :return: Two 1D tensors of representing x and y coordinates of the points.
        :rtype: tuple[Tensor, Tensor]
        """
        x, y = self._sample_unipolar(n_angle, n_radius)
        return self._discard_invalid(x, y)

    def sample_diameter(self, n: int = 64, theta: float = 0.) -> tuple[Ts, Ts]:
        """
        Samples points on diameter line segments of this aperture.
        Polar angle of the line is given by ``theta``.

        :param int n: Number of points.
        :param float theta: Polar angle of the line.
        :return: Two 1D tensors representing x and y coordinates of the points.
        :rtype: tuple[Tensor, Tensor]
        """
        x, y = self._sample_diameter(n, theta)
        return self._discard_invalid(x, y)

    def _sample_random(self, n):
        w, h = self.max_width(), self.max_height()
        x = self.rand(n) * w - w / 2
        y = self.rand(n) * h - h / 2
        return x, y

    def _sample_rect(self, n):
        n = ty.size2d(n)
        w, h = self.max_width(), self.max_height()
        x = self.linspace(-w / 2, w / 2, n[1])
        y = self.linspace(-h / 2, h / 2, n[0])
        x, y = torch.meshgrid(x, y, indexing='ij')
        x, y = x.flatten(), y.flatten()
        return x, y

    def _sample_unipolar(self, n_angle, n_radius):
        zero = self.new_tensor([0.])
        r = self.linspace(0, 1, n_radius + 1) * self.max_radius()
        r = [r[i].expand(i * n_angle) for i in range(1, n_radius + 1)]  # n_t*n_r*(n_r+1)/2
        r = torch.cat([zero] + r)  # n_t*n_r*(n_r+1)/2+1
        t = [
            self.arange(i * n_angle) / (n_angle * i) * (2 * torch.pi)
            for i in range(1, n_radius + 1)
        ]
        t = torch.cat([zero] + t)
        return r * t.cos(), r * t.sin()

    def _sample_diameter(self, n, theta):
        r = self.linspace(-1, 1, n) * self.max_radius()
        theta = base.Angle.default_to(theta, 'rad')
        theta = self.new_tensor(theta)
        x, y = r * theta.cos(), r * theta.sin()
        return x, y

    def _discard_invalid(self, x: Ts, y: Ts) -> tuple[Ts, Ts]:
        valid = self.evaluate(x, y)
        x, y = x[valid], y[valid]
        return x, y


class CircularAperture(BoundedAperture):
    """
    Circular aperture with radius :attr:`radius`.

    :param radius: Radius of the aperture. Must be finite.
        Default: :data:`dnois.conf.default_aperture_radius`.
    :type radius: float | Tensor
    """

    def __init__(self, radius: ty.Scalar = None):
        super().__init__()
        if radius is None:
            radius = _default_aperture_size()

        self.register_parameter('radius', None)
        radius = ty.scalar(radius, dtype=torch.get_default_dtype())
        #: Radius of the aperture.
        self.radius: nn.Parameter = nn.Parameter(radius, False)

    def extra_repr(self) -> str:
        return f'radius={base.Length.fmt(self.radius.item())}'

    def evaluate(self, x: Ts, y: Ts) -> torch.BoolTensor:
        return ty.cast(torch.BoolTensor, x.square() + y.square() < self._detection_radius().square())

    def max_radius(self) -> Ts:
        return self.radius

    def max_width(self) -> Ts:
        return self.diameter

    def pass_ray(self, ray: BatchedRay) -> torch.BoolTensor:
        return ty.cast(torch.BoolTensor, ray.r2 < self._detection_radius().square())

    def sample_random(self, n: int) -> tuple[Ts, Ts]:
        """
        Similar to :meth:`BoundedAperture.sample_random`, but guarantees
        the number of returned points.
        """
        t = self.rand(n) * (2 * torch.pi)
        r = self.rand(n)
        r = r.sqrt()
        r = r * self.radius
        return r * t.cos(), r * t.sin()

    def sample_unipolar(self, n_radius: int = 6, n_angle: int = 6) -> tuple[Ts, Ts]:
        """
        Similar to :meth:`BoundedAperture.sample_unipolar`, but guarantees
        the number of returned points.
        """
        return self._sample_unipolar(n_angle, n_radius)

    def sample_diameter(self, n: int = 64, theta: float = 0.) -> tuple[Ts, Ts]:
        """
        Similar to :meth:`BoundedAperture.sample_diameter`, but guarantees
        the number of returned points.
        """
        return self._sample_diameter(n, theta)

    def to_dict(self, keep_tensor=True) -> dict[str, ty.Any]:
        d = super().to_dict(keep_tensor)
        d['radius'] = self._attr2dictitem('radius', keep_tensor)
        return d

    @property
    def diameter(self):
        """Diameter of the aperture.\n\n:type: 0D Tensor"""
        return self.radius * 2

    @property
    def r(self):
        """Alias for :attr:`.radius`."""
        return self.radius

    @property
    def d(self):
        """Alias for :attr:`.diameter`."""
        return self.diameter

    def _detection_radius(self) -> Ts:
        return self.radius * (1 + conf.detection_radius_eps)


class AnnularAperture(BoundedAperture):
    """
    Annular aperture with an inner radius and an outer one.
    Only rays falling within the inner and outer radius are considered as valid.

    :param inner_r: Inner radius. Must be smaller than ``outer_r``. Default : 0.
    :type inner_r: float | Tensor
    :param outer_r: Outer radius. Must be finite.
        Default: :data:`dnois.conf.default_aperture_radius`.
    :type outer_r: float | Tensor
    """

    def __init__(self, inner_r: ty.Scalar = 0., outer_r: ty.Scalar = None):
        super().__init__()
        if outer_r is None:
            outer_r = _default_aperture_size()

        self.register_parameter('inner_r', None)
        self.register_parameter('outer_r', None)
        r1 = ty.scalar(inner_r, dtype=torch.get_default_dtype())
        r2 = ty.scalar(outer_r, dtype=torch.get_default_dtype())
        if r1.item() < 0 or r2.item() < 0:
            raise ValueError('inner_r and outer_r must be non-negative')
        if r1.item() >= r2.item():
            raise ValueError('inner_r must be smaller than outer_r')

        self.r1: nn.Parameter = nn.Parameter(r1, False)  #: Inner radius.
        self.r2: nn.Parameter = nn.Parameter(r2, False)  #: Outer radius.

    def extra_repr(self) -> str:
        return f'R1={base.Length.fmt(self.r1.item())}, R2={base.Length.fmt(self.r2.item())}'

    def evaluate(self, x: Ts, y: Ts) -> torch.BoolTensor:
        r2 = x.square() + y.square()
        valid = (r2 > self._detection_r1().square()) & (r2 < self._detection_r2().square())
        return ty.cast(torch.BoolTensor, valid)

    def max_radius(self) -> Ts:
        return self.r2

    def max_width(self) -> Ts:
        return self.d2

    def pass_ray(self, ray: BatchedRay) -> torch.BoolTensor:
        valid = (ray.r2 > self._detection_r1().square()) & (ray.r2 < self._detection_r2().square())
        return ty.cast(torch.BoolTensor, valid)

    def sample_random(self, n: int) -> tuple[Ts, Ts]:
        """
        Similar to :meth:`BoundedAperture.sample_random`, but guarantees
        the number of returned points.
        """
        t = self.rand(n) * (2 * torch.pi)
        r = self.rand(n) * (self.r2.square() - self.r1.square()) + self.r1.square()
        r = r.sqrt()
        r = r * self.radius
        return r * t.cos(), r * t.sin()

    def sample_diameter(self, n: int = 64, theta: float = 0.) -> tuple[Ts, Ts]:
        """
        Similar to :meth:`BoundedAperture.sample_diameter`, but guarantees
        the number of returned points.
        """
        if n % 2:
            raise ValueError(f'n must be even, but got {n}')
        r = self.linspace(0, 1, n // 2) * (self.r2 - self.r1) + self.r1
        r = torch.cat([-r.flip(0), r])
        theta = base.Angle.default_to(theta, 'rad')
        theta = self.new_tensor(theta)
        return r * theta.cos(), r * theta.sin()

    def to_dict(self, keep_tensor=True) -> dict[str, ty.Any]:
        d = super().to_dict(keep_tensor)
        d['inner_r'] = self._attr2dictitem('r1', keep_tensor)
        d['outer_r'] = self._attr2dictitem('r2', keep_tensor)
        return d

    @property
    def min_r(self) -> nn.Parameter:
        """Alias for :attr:`.r1`."""
        return self.r1

    @property
    def max_r(self) -> nn.Parameter:
        """Alias for :attr:`.r2`."""
        return self.r2

    @property
    def d1(self) -> Ts:
        """The inner diameter.\n\n:type: Tensor"""
        return 2 * self.r1

    @property
    def d2(self) -> Ts:
        """The outer diameter.\n\n:type: Tensor"""
        return 2 * self.r2

    def _detection_r1(self) -> Ts:
        return self.r1 * (1 - conf.detection_radius_eps)

    def _detection_r2(self) -> Ts:
        return self.r2 * (1 + conf.detection_radius_eps)


class RectangularAperture(BoundedAperture):
    """
    Rectangular aperture with widths in x and y directions.

    :param width_x: Width in x direction. Must be finite.
        Default: double of :data:`dnois.conf.default_aperture_radius`.
    :type width_x: float | Tensor
    :param width_y: Width in y direction. Must be finite.
        Default: equal to ``width_x``.
    :type width_y: float | Tensor
    """

    def __init__(self, width_x: ty.Scalar = None, width_y: ty.Scalar = None):
        super().__init__()
        if width_x is None:
            width_x = _default_aperture_size() * 2
        if width_y is None:
            width_y = width_x

        self.register_parameter('width_x', None)
        self.register_parameter('width_y', None)
        w1 = ty.scalar(width_x, dtype=torch.get_default_dtype())
        w2 = ty.scalar(width_y, dtype=torch.get_default_dtype())
        if w1.item() <= 0 or w2.item() <= 0:
            raise ValueError('width_x and width_y must be positive')

        self.width_x: nn.Parameter = nn.Parameter(w1, False)
        self.width_y: nn.Parameter = nn.Parameter(w2, False)

    def extra_repr(self) -> str:
        return f'width_x={base.Length.fmt(self.width_x.item())}, width_y={base.Length.fmt(self.width_y.item())}'

    def max_radius(self) -> Ts:
        return torch.sqrt(self.width_x.square() + self.width_y.square()) / 2

    def max_width(self) -> Ts:
        return self.width_x

    def max_height(self) -> Ts:
        return self.width_y

    def evaluate(self, x: Ts, y: Ts) -> torch.BoolTensor:
        ratio = (1 + conf.detection_radius_eps) / 2
        return ty.cast(torch.BoolTensor, torch.logical_and(
            x.abs() <= self.width_x * ratio, y.abs() <= self.width_y * ratio
        ))

    def sample_random(self, n: int) -> tuple[Ts, Ts]:
        """
        Similar to :meth:`BoundedAperture.sample_random`, but guarantees
        the number of returned points.
        """
        return self._sample_random(n)

    def sample_rect(self, n: ty.Size2d) -> tuple[Ts, Ts]:
        """
        Similar to :meth:`BoundedAperture.sample_rect`, but guarantees
        the number of returned points.
        """
        return self._sample_rect(n)

    def to_dict(self, keep_tensor=True) -> dict[str, ty.Any]:
        d = super().to_dict(keep_tensor)
        d['width_x'] = self._attr2dictitem('width_x', keep_tensor)
        d['width_y'] = self._attr2dictitem('width_y', keep_tensor)
        return d
