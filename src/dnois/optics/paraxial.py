import abc
import dataclasses
import math

import torch

from . import _func
from .. import base, utils
from ..base.typing import Numeric, Ts, Self, cast

__all__ = [
    'ConjugatePointFeature',
    'FiniteParaxialSystem',
    'InfiniteParaxialSystem',
    'ParaxialSystem',
]


def _determine(principal: Numeric, focal: Numeric, fl: Numeric, obj: bool):
    if fl is None:
        if principal is None or focal is None:
            raise ValueError('principal and focal must be provided if fl is not')
        fl = principal - focal if obj else focal - principal
    else:
        if focal is not None:
            if principal is None:
                principal = focal + fl if obj else focal - fl
            else:
                raise ValueError('principal and focal cannot be provided if fl is given')
    return principal, fl


@dataclasses.dataclass
class ConjugatePointFeature:
    r"""
    A dataclass to store various features of the conjugate points:

    - Object distance :math:`s`.
    - Image distance :math:`s'=\frac{f's}{s-f}`.
    - Transverse magnification :math:`\frac{f}{f-s}`.
    - Longitudinal magnification :math:`-\frac{ff'}{(s-f)^2}`.
    - Angular magnification :math:`\frac{f-s}{f'}`.

    where :math:`f` and :math:`f'` are the focal lengths in object and image space.
    """
    obj_d: Numeric = None  #: Object distance.
    img_d: Numeric = None  #: Image distance.
    transverse: Numeric = None  #: Transverse magnification.
    longitudinal: Numeric = None  #: Longitudinal magnification.
    angular: Numeric = None  #: Angular magnification.


class ParaxialSystem(metaclass=abc.ABCMeta):
    """
    Model of paraxial (or ideal, Gaussian) optical system. Its main properties include
    the positions of two principal points. Principal points can also be left unspecified.
    Their positions are defined as their coordinates on optical axis, where smaller
    coordinate implies object space.

    See :doc:`/content/guide/optics/paraxial` for more information.

    :param principal1: Position of object principal point. Default: not specified.
    :param principal2: Position of image principal point. Default: not specified.
    """

    fl1: Numeric
    fl2: Numeric

    def __init__(self, principal1: Numeric = None, principal2: Numeric = None):
        self.principal1 = principal1
        self.principal2 = principal2

    def __repr__(self):
        return f'{self.__class__.__name__}({self._repr_fields()})'

    @abc.abstractmethod
    def composite(self, other: 'ParaxialSystem', delta: Numeric = None, d: Numeric = None) -> Self:
        """
        Composite two paraxial systems. To compute the focal lengths of resulted system,
        their distance must be determined, which can be specified by the distance between their
        focal points (``delta``) or their principal points (``d``). If both systems have
        definite principal points, these two parameters cannot be given.

        :param ParaxialSystem other: Another paraxial system.
        :param delta: Distance between the two focal points.
        :param d: Distance between the two principal points.
        :return: A new paraxial system.
        :rtype: ParaxialSystem
        """
        pass

    @abc.abstractmethod
    def flip(self) -> Self:
        """
        Flip the paraxial system.

        :return: A new paraxial system.
        :rtype: ParaxialSystem
        """
        pass

    def imgd(self, obj_d: Numeric) -> Numeric:
        """Call :func:`dnois.optics.imgd` with focal lengths of this system."""
        return _func.imgd(obj_d, self.fl1, self.fl2)

    def objd(self, img_d: Numeric) -> Numeric:
        """Call :func:`dnois.optics.objd` with focal lengths of this system."""
        return _func.objd(img_d, self.fl1, self.fl2)

    def feature(
        self,
        obj_d: Numeric,
        img_d: bool = True,
        transverse: bool = True,
        longitudinal: bool = True,
        angular: bool = True,
    ) -> ConjugatePointFeature:
        """
        Compute various features of the conjugate points at a given object distance.

        .. seealso::
            :class:`ConjugatePointFeature`

        :param obj_d: Object distance.
        :param bool img_d: Whether to compute image distance. Default: ``True``.
        :param bool transverse: Whether to compute transverse magnification. Default: ``True``.
        :param bool longitudinal: Whether to compute longitudinal magnification. Default: ``True``.
        :param bool angular: Whether to compute angular magnification. Default: ``True``.
        :return: A :class:`ConjugatePointFeature` object.
        :rtype: ConjugatePointFeature
        """
        f_minus_s = self.fl1 - obj_d
        computation = utils.InfinityCond(
            [  # expressions when object distance is finite
                lambda _: -self.fl2 * obj_d / f_minus_s if img_d else None,
                lambda _: self.fl1 / f_minus_s if transverse else None,
                lambda _: -self.fl1 * self.fl2 / f_minus_s ** 2 if longitudinal else None,
                lambda _: f_minus_s / self.fl2 if angular else None,
            ],
            [  # expressions when object distance is infinite
                lambda _: self.fl2 if img_d else None,
                lambda _: 0. if transverse else None,
                lambda _: 0. if longitudinal else None,
                lambda _: float('inf') if angular else None,
            ]
        )
        f = ConjugatePointFeature(obj_d)
        f.img_d, f.transverse, f.longitudinal, f.angular = computation(obj_d)
        return f

    def img_point(self, obj_point: Ts) -> Ts:
        """
        Compute the conjugate point of a given point in object space.
        The point is specified by its 3D coordinate. Note that the optical axis is z-axis,
        while the directions of x-axis and y-axis do not matter.

        :param Tensor obj_point: The point in object space. A tensor whose size in last dimension
            is either 3 where the first two indicates x- and y-coordinates,
            or 2 where the first indicates lateral coordinates.
        :return: The conjugate point in image space. Its shape is broadcast from that of
            ``obj_point`` and some features of ``self``.
        :rtype: Tensor
        """
        if obj_point.size(-1) == 3:  # noqa
            lateral, z = obj_point[..., :2], obj_point[..., 2]
        elif obj_point.size(-1) == 2:
            lateral, z = obj_point[..., [0]], obj_point[..., 1]
        else:
            raise ValueError(f'Size of the last dimension of obj_point must be 2 or 3, got {obj_point.size(-1)}')
        obj_d = self.principal1 - z
        feat = self.feature(obj_d, longitudinal=False, angular=False)
        z_img = self.principal2 + feat.img_d
        lateral = lateral * feat.transverse
        return torch.cat([lateral, z_img], dim=-1)

    def obj_point(self, img_point: Ts) -> Ts:
        """Similar to :meth:`.img_point` but computes object point from image point."""
        if img_point.size(-1) == 3:  # noqa
            lateral, z = img_point[..., :2], img_point[..., 2]
        elif img_point.size(-1) == 2:
            lateral, z = img_point[..., [0]], img_point[..., 1]
        else:
            raise ValueError(f'Size of the last dimension of img_point must be 2 or 3, got {img_point.size(-1)}')
        img_d = z - self.principal2
        obj_d = self.objd(img_d)
        feat = self.feature(obj_d, img_d=False, longitudinal=False, angular=False)
        z_obj = self.principal1 - obj_d
        lateral = lateral / feat.transverse
        return torch.cat([lateral, z_obj], dim=-1)

    @classmethod
    def from_refractive_interface(
        cls, roc: Numeric, n1: Numeric, n2: Numeric, location: Numeric = None
    ) -> Self:
        """
        Construct a paraxial system from a refractive interface between two media given its radius of curvature.
        See :doc:`/content/guide/optics/paraxial` for the rule about the sign of radius of curvature.

        :param roc: Radius of curvature of the interface.
        :param n1: Refractive index of the medium before the interface.
        :param n2: Refractive index of the medium after the interface.
        :param location: Location of the interface which serves as the principal points. Default: not specified.
        :return: A paraxial system.
        :rtype: ParaxialSystem
        """
        dn = n2 - n1
        if isinstance(roc, float):
            if math.isinf(roc):
                return InfiniteParaxialSystem(n2 / n1, location, location)
            else:
                inv_diopter = roc / dn
                return FiniteParaxialSystem(location, location, None, None, n1 * inv_diopter, n2 * inv_diopter)
        elif torch.is_tensor(roc):
            if roc.isinf().all():
                return InfiniteParaxialSystem(n2 / n1, location, location)
            elif not roc.isinf().any():
                inv_diopter = roc / dn
                return FiniteParaxialSystem(location, location, None, None, n1 * inv_diopter, n2 * inv_diopter)
            else:
                raise RuntimeError('roc must be either all infinite or all finite')
        else:
            raise TypeError(f'roc must be a float or a tensor, got {type(roc).__name__}')

    @classmethod
    def from_reflective_interface(cls, roc: Numeric, location: Numeric = None) -> Self:
        """
        Construct a paraxial system from a reflective interface between two media given its radius of curvature.
        See :doc:`/content/guide/optics/paraxial` for the rule about the sign of radius of curvature.

        :param roc: Radius of curvature of the interface.
        :param location: Location of the interface which serves as the principal points. Default: not specified.
        :return: A paraxial system.
        :rtype: ParaxialSystem
        """
        return cls.from_refractive_interface(roc, 1, -1, location)

    def _determine_distance(self, other: 'ParaxialSystem', d: Numeric = None) -> Numeric:
        p2a, p1b = self.principal2, other.principal1
        if p2a is not None and p1b is not None:
            if d is not None:
                raise ValueError('d cannot be provided because the distance is determined')
            else:
                d = p1b - p2a
        if d is None:
            raise ValueError('d must be provided')
        return d

    def _repr_fields(self) -> str:
        p_texts = []
        if self.principal1 is not None:
            p_texts.append(f'principal1={base.Length.fmt(self.principal1)}')
        if self.principal2 is not None:
            p_texts.append(f'principal2={base.Length.fmt(self.principal2)}')
        return ", ".join(p_texts)


class FiniteParaxialSystem(ParaxialSystem):
    """
    An implementation of :class:`ParaxialSystem` whose focal lengths are both finite.

    In the constructor either the position of a focal point or the corresponding focal length must be
    specified but cannot be both.

    See :class:`ParaxialSystem` for descriptions of more parameters.

    :param focal1: The position of focal point in object space.
    :param focal2: The position of focal point in image space.
    :param fl1: Focal length in object space.
    :param fl2: Focal length in image space.
    """

    def __init__(
        self,
        principal1: Numeric = None,
        principal2: Numeric = None,
        focal1: Numeric = None,
        focal2: Numeric = None,
        fl1: Numeric = None,
        fl2: Numeric = None,
    ):
        p1, fl1 = _determine(principal1, focal1, fl1, True)
        p2, fl2 = _determine(principal2, focal2, fl2, False)
        super().__init__(p1, p2)
        self.fl1, self.fl2 = fl1, fl2

    def composite(self, other: ParaxialSystem, delta: Numeric = None, d: Numeric = None) -> ParaxialSystem:
        if isinstance(other, FiniteParaxialSystem):
            return self._composite_finite(other, delta, d)
        elif isinstance(other, InfiniteParaxialSystem):
            return self._composite_infinite(other, d)
        else:
            raise TypeError(f'Composition between {type(self).__name__} and {type(other).__name__} is not supported')

    def flip(self) -> Self:
        return FiniteParaxialSystem(self.principal2, self.principal1, fl1=-self.fl2, fl2=-self.fl1)

    @property
    def focal1(self) -> Numeric | None:
        """Position of the focal point in object space. ``None`` if it is not specified."""
        return None if self.principal1 is None else self.principal1 - self.fl1

    @focal1.setter
    def focal1(self, value):
        self.principal1 = value + self.fl1

    @property
    def focal2(self) -> Numeric | None:
        """Position of the focal point in image space. ``None`` if it is not specified."""
        return None if self.principal2 is None else self.principal2 + self.fl2

    @focal2.setter
    def focal2(self, value):
        self.principal2 = value - self.fl2

    def _repr_fields(self) -> str:
        p_texts = super()._repr_fields().split(', ')
        if self.fl1 is not None:
            p_texts.append(f'focal1={base.Length.fmt(self.focal1)}')
            p_texts.append(f'fl1={base.Length.fmt(self.fl1)}')
        if self.fl2 is not None:
            p_texts.append(f'focal2={base.Length.fmt(self.focal2)}')
            p_texts.append(f'fl2={base.Length.fmt(self.fl2)}')
        return ", ".join(p_texts)

    def _composite_finite(
        self, other: 'FiniteParaxialSystem', delta: Numeric = None, d: Numeric = None
    ) -> 'FiniteParaxialSystem':
        focal2a, focal1b = self.focal2, other.focal1
        if focal2a is not None and focal1b is not None:
            if not (delta is None and d is None):
                raise ValueError('delta and d cannot be provided because the distance is determined')
            else:
                delta = cast(Numeric, focal1b) - cast(Numeric, focal2a)

        if delta is None:
            if d is None:
                raise ValueError('Either delta or d must be provided')
            delta = d - self.fl2 - other.fl1
        else:
            if d is not None:
                raise ValueError('delta cannot be provided if d is')
            d = self.fl2 + delta + other.fl1

        if self.principal1 is None:
            p1 = None
        else:
            p1 = self.principal1 - self.fl1 * d / delta
        if other.principal2 is None:
            p2 = None
        else:
            p2 = other.principal2 + other.fl2 * d / delta
        return FiniteParaxialSystem(p1, p2, None, None, -self.fl1 * other.fl1 / delta, -self.fl2 * other.fl2 / delta)

    def _composite_infinite(self, other: 'InfiniteParaxialSystem', d: Numeric = None) -> 'FiniteParaxialSystem':
        d = self._determine_distance(other, d)
        if other.principal2 is None:
            p2 = None
        else:
            p2 = other.principal2 - d * other.focal_ratio
        fl1, fl2 = self.fl1, self.fl2 * other.focal_ratio
        return FiniteParaxialSystem(self.principal1, p2, None, None, fl1, fl2)


class InfiniteParaxialSystem(ParaxialSystem):
    """
    An implementation of :class:`ParaxialSystem` whose focal lengths are both infinite.
    However, the ratio of the image focal length to the object one is finite.

    See :class:`ParaxialSystem` for descriptions of more parameters.

    :param focal_ratio: Ratio of the image focal length to the object one.
    """

    def __init__(
        self,
        focal_ratio: Numeric,
        principal1: Numeric = None,
        principal2: Numeric = None,
    ):
        super().__init__(principal1, principal2)
        self.focal_ratio = focal_ratio

    def composite(self, other: ParaxialSystem, delta: Numeric = None, d: Numeric = None) -> ParaxialSystem:
        if delta is not None:
            raise ValueError('delta cannot be provided for infinite paraxial system')
        d = self._determine_distance(other, d)
        if isinstance(other, FiniteParaxialSystem):
            return self._composite_finite(other, d)
        elif isinstance(other, InfiniteParaxialSystem):
            return self._composite_infinite(other, d)
        else:
            raise TypeError(f'Composition between {type(self).__name__} and {type(other).__name__} is not supported')

    def flip(self) -> Self:
        return InfiniteParaxialSystem(1 / self.focal_ratio, self.principal2, self.principal1)

    def imgd(self, obj_d: Numeric) -> Numeric:
        return obj_d * -self.focal_ratio

    def objd(self, img_d: Numeric) -> Numeric:
        return img_d / -self.focal_ratio

    def feature(
        self,
        obj_d: Numeric,
        img_d: bool = True,
        transverse: bool = True,
        longitudinal: bool = True,
        angular: bool = True,
    ) -> ConjugatePointFeature:
        kwargs = {}
        if img_d:
            kwargs['img_d'] = self.imgd(obj_d)
        if transverse:
            kwargs['transverse'] = torch.ones_like(obj_d) if torch.is_tensor(obj_d) else 1.
        if longitudinal:
            if torch.is_tensor(obj_d):
                kwargs['longitudinal'] = -self.focal_ratio * torch.ones_like(obj_d)
            else:
                kwargs['longitudinal'] = -self.focal_ratio
        if angular:
            if torch.is_tensor(obj_d):
                kwargs['angular'] = 1 / self.focal_ratio * torch.ones_like(obj_d)
            else:
                kwargs['angular'] = 1 / self.focal_ratio
        return ConjugatePointFeature(obj_d, **kwargs)

    @property
    def fl1(self):
        return float('inf')

    @property
    def fl2(self):
        return float('inf') * self.focal_ratio

    def _repr_fields(self) -> str:
        p_texts = super()._repr_fields().split(', ')
        p_texts.append(f'focal_ratio={self.focal_ratio}')
        return ", ".join(p_texts)

    def _composite_finite(self, other: FiniteParaxialSystem, d: Numeric) -> FiniteParaxialSystem:
        if self.principal1 is None:
            p1 = None
        else:
            p1 = self.principal1 + d / self.focal_ratio
        fl1, fl2 = other.fl1 / self.focal_ratio, other.fl2
        return FiniteParaxialSystem(p1, other.principal2, None, None, fl1, fl2)

    def _composite_infinite(self, other: 'InfiniteParaxialSystem', d: Numeric) -> 'InfiniteParaxialSystem':
        if other.principal2 is None:
            p2 = None
        else:
            p2 = other.principal2 - d * other.focal_ratio
        return InfiniteParaxialSystem(self.focal_ratio * other.focal_ratio, self.principal1, p2)
