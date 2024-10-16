import abc
import collections.abc

import torch
from torch import nn

from .ray import BatchedRay
from ... import mt, base
from ...base.typing import (
    Sequence, Ts, Any, Callable, Scalar, Self, Size2d,
    scalar, size2d, cast
)

__all__ = [
    'NT_EPSILON',
    'NT_MAX_ITERATION',
    'NT_THRESHOLD',
    'NT_THRESHOLD_STRICT',
    'NT_UPDATE_BOUND',

    'Aperture',
    'BatchedRay',
    'CircularAperture',
    'CircularSurface',
    'Context',
    'Surface',
    'SurfaceList',
]

NT_MAX_ITERATION: int = 10
NT_THRESHOLD: float = 20e-9
NT_THRESHOLD_STRICT: float = 20e-9
NT_UPDATE_BOUND: float = 5.
NT_EPSILON: float = 1e-9

SAMPLE_LIMIT: float = 1 - 1e-4
EDGE_CUTTING: float = 1 - 1e-5


def _dist_transform(x: Ts, curve: Callable[[Ts], Ts]) -> Ts:
    magnitude = curve(x.abs())
    return magnitude.copysign(x)


class Context:
    """
    A class representing the context of a :py:class:`~Surface` in a lens group. As component of the
    lens group, a surface does not hold the reference to the group but can access
    the information that depends on other surfaces in it via this class.
    Every surface contained in a lens group has a related context object.
    If it is not contained in any group, its context attribute is ``None``.

    :param Surface surface: The host surface that this context belongs to.
    :param SurfaceList lens_group: The lens group containing the surface.
    """

    def __init__(self, surface: 'Surface', lens_group: 'SurfaceList'):
        self.surface: 'Surface' = surface  #: The host surface that this context belongs to.
        self.lens_group: 'SurfaceList' = lens_group  #: The lens group containing the surface.

    @property
    def index(self) -> int:
        """
        The index of the host surface in the lens group.

        :type: int
        """
        return self.lens_group.index(self.surface)

    @property
    def material_before(self) -> mt.Material:
        """
        :py:class:`~dnois.mt.Material` object before ths host surface.

        :type: :py:class:`~dnois.mt.Material`
        """
        idx = self.index
        if idx == 0:
            return self.lens_group.env_material
        return self.lens_group[idx - 1].material

    @property
    def z(self) -> Ts:
        """
        The z-coordinate of the related surface's baseline. A 0D tensor.

        :type: Tensor
        """
        idx = self.index
        if idx == 0:
            return self.surface.distance.new_tensor(0.)
        z = self.lens_group[0].distance
        for s in self.lens_group[1:idx]:
            z = z + s.distance
        return z

    def _check_available(self):
        if self.surface in self.lens_group:
            return
        raise RuntimeError(
            'The surface is not contained in the lens group referenced by its context object. '
            'This may be because the surface has been removed from the lens group.')


class Aperture(base.TensorContainerMixIn, nn.Module, metaclass=abc.ABCMeta):
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

    @abc.abstractmethod
    def sample_random(self, n: int) -> tuple[Ts, Ts]:
        """
        Returns ``n`` points randomly sampled on this aperture.

        :param int n: Number of points.
        :return: Two 1D tensors of length ``n``, representing x and y coordinates of the points.
        :rtype: tuple[Tensor, Tensor]
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


class CircularAperture(Aperture):
    """
    Circular aperture with radius :attr:`radius`.

    :param radius: Radius of the aperture.
    :type radius: float | Tensor
    """

    def __init__(self, radius: Scalar):
        super().__init__()
        if radius <= 0:
            raise ValueError('radius must be positive')
        self.radius: Ts = scalar(radius)  #: Radius of the aperture.

    def evaluate(self, x: Ts, y: Ts) -> torch.BoolTensor:
        return cast(torch.BoolTensor, x.square() + y.square() <= self.radius.square())

    def pass_ray(self, ray: BatchedRay) -> torch.BoolTensor:
        return cast(torch.BoolTensor, ray.r2 <= self.radius.square())

    def sample_random(self, n: int, sampling_curve: Callable[[Ts], Ts] = None) -> tuple[Ts, Ts]:
        r"""
        Returns ``n`` points randomly sampled on this aperture. An optional ``sampling_curve``
        (denoted by :math:`\Gamma`) can be specified to control the distribution of
        radial distance: :math:`r=\Gamma(t)` where :math:`t` is drawn uniformly from :math:`[0,1]`.

        :param int n: Number of points.
        :param sampling_curve: Sampling curve :math:`\Gamma(t)`. Default: :math:`\sqrt{t}`.
        :type sampling_curve: Callable[[Tensor], Tensor]
        :return: Two 1D tensors of length ``n``, representing x and y coordinates of the points.
        :rtype: tuple[Tensor, Tensor]
        """
        t = torch.rand(n, device=self.device, dtype=self.dtype) * (2 * torch.pi)
        r = torch.rand(n, device=self.device, dtype=self.dtype)
        if sampling_curve is not None:
            r = sampling_curve(r)
        else:
            r = r.sqrt()
        r = r * self.radius
        return r * t.cos(), r * t.sin()

    def sample_unipolar(self, n: Size2d) -> tuple[Ts, Ts]:
        r"""
        Samples points on this aperture in a unipolar manner. Specifically, the aperture
        is divided into :math:`N_r` rings with equal widths and points are sampled on the
        outer edge of each ring. The first ring contains :math:`N_\theta` points, the second
        contains :math:`2N_\theta` points ... and so on, plus a point at center.
        Thus, there are :math:`N_\theta N_r(N_r+1)/2+1` points in total.

        :param n: A pair of int representing :math:`(N_r,N_\theta)`.
        :type n: int | tuple[int, int]
        :return: Two 1D tensors of representing x and y coordinates of the points.
        :rtype: tuple[Tensor, Tensor]
        """
        n = size2d(n)
        zero = torch.tensor(0., dtype=self.dtype, device=self.device)
        r = torch.linspace(0, self.radius * EDGE_CUTTING, n[0] + 1, device=self.device, dtype=self.dtype)
        r = [r[i].expand(i * n[1]) for i in range(1, n[0] + 1)]  # n[1]*n[0]*(n[0]+1)/2
        r = torch.cat([zero] + r)  # n[1]*n[0]*(n[0]+1)/2+1
        t = [
            torch.arange(i * n[1], device=self.device, dtype=self.dtype) / (n[1] * i) * (2 * torch.pi)
            for i in range(1, n[0] + 1)
        ]
        t = torch.cat([zero] + t)
        return r * t.cos(), r * t.sin()

    def _delegate(self) -> Ts:
        return self.radius


class Surface(base.TensorContainerMixIn, nn.Module, metaclass=abc.ABCMeta):
    r"""
    Base class for optical surfaces in a group of lens.
    The position of a surface in a lens group is specified by a single z-coordinate,
    which is called its *baseline*. Baseline of the first surface is 0.

    The geometric shape of a surface is described by an equation
    :math:`z=h(x,y)+z_\text{baseline}`, which has different forms
    for each surface type. The function :math:`h`, called *surface function*,
    is a 2D function of lateral coordinates :math:`(x,y)` which satisfies :math:`h(0,0)=0`.
    Note that the surface function also depends on the parameters of the surface implicitly.

    To ensure that a ray propagating along z-axis must have an intersection with
    the surface, an extended surface function (see :py:meth:`~h_extended`)
    is computed to find the intersection. Normally, the definition domain of
    surface function covers the aperture so an extended surface does not
    affect actual surface. If it cannot cover the aperture, however, the
    actual surface will be extended, which is usually undesired.
    At present a warning will be issued if that case is detected.

    This is subclass of :py:class:`torch.nn.Module`.

    .. note::

        All the quantities with length dimension is represented in meters.
        They include value of coordinates, optical path length and wavelength, etc.

    :param material: Material following the surface. Either a :py:class:`~dnois.mt.Material`
        instance or a str representing the name of a registered material.
    :type material: :py:class:`~dnois.mt.Material` or str
    :param distance: Distance between the surface and the next one.
    :type distance: float or Tensor
    :param Aperture aperture: :class:`Aperture` of this surface.
    :param dict newton_config: Configuration for Newton's method.
        See :ref:`configuration_for_newtons_method` for details.
    """

    def __init__(
        self,
        material: mt.Material | str,
        distance: Scalar,
        aperture: Aperture,
        newton_config: dict[str, Any] = None
    ):
        super().__init__()
        distance = scalar(distance)
        if distance.item() < 0:
            raise ValueError('distance must not be negative')
        if newton_config is None:
            newton_config = {}
        #: Material following the surface.
        self.material: mt.Material = material if isinstance(material, mt.Material) else mt.get(material)
        #: Distance between the surface and the next one.
        #: This is an optimizable parameter by default.
        self.distance: nn.Parameter = nn.Parameter(distance)
        #: :class:`Aperture` of this surface.
        self.aperture: Aperture = aperture
        #: The context object of the surface in a lens group.
        #: This is created by the lens group object containing the surface.
        self.context: Context | None = None

        self._nt_max_iteration = newton_config.get('max_iteration', NT_MAX_ITERATION)
        self._nt_threshold = newton_config.get('threshold', NT_THRESHOLD)
        self._nt_threshold_strict = newton_config.get('threshold_strict', NT_THRESHOLD_STRICT)
        self._nt_update_bound = newton_config.get('update_bound', NT_UPDATE_BOUND)
        self._nt_epsilon = newton_config.get('epsilon', NT_EPSILON)

    @abc.abstractmethod
    def h(self, x: Ts, y: Ts) -> Ts:
        r"""
        Computes surface function :math:`h(x,y)`.

        :param Tensor x: x coordinate.
        :param Tensor y: y coordinate.
        :return: Corresponding value of the surface function.
        :rtype: Tensor
        """
        pass

    @abc.abstractmethod
    def h_grad(self, x: Ts, y: Ts) -> tuple[Ts, Ts]:
        r"""
        Computes the partial derivatives of surface function
        :math:`\pfrac{h(x,y)}{x}` and :math:`\pfrac{h(x,y)}{y}`.

        :param Tensor x: x coordinate.
        :param Tensor y: y coordinate.
        :return: Corresponding value of two partial derivatives.
        :rtype: tuple[Tensor, Tensor]
        """
        pass

    @abc.abstractmethod
    def h_extended(self, x: Ts, y: Ts) -> Ts:
        r"""
        Computes extended surface function:

        .. math::

            \tilde{h}(x,y)=\left\{\begin{array}{ll}
                h(x,y) & \text{if}(x,y)\in\text{dom} h,\\
                \text{extended value} & \text{else}
            \end{array}\right.

        :param Tensor x: x coordinate.
        :param Tensor y: y coordinate.
        :return: Corresponding value of extended surface function.
        :rtype: Tensor
        """
        pass

    @abc.abstractmethod
    def h_grad_extended(self, x: Ts, y: Ts) -> tuple[Ts, Ts]:
        r"""
        Computes partial derivatives of extended surface function:
        :math:`\pfrac{\tilde{h}(x,y)}{x}` and :math:`\pfrac{\tilde{h}(x,y)}{y}`.
        See :py:meth:`~h_extended`.

        :param Tensor x: x coordinate.
        :param Tensor y: y coordinate.
        :return: Corresponding value of two partial derivatives.
        :rtype: tuple[Tensor, Tensor]
        """
        pass

    def forward(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        """
        Returns the refracted rays of a group of incident rays ``ray``.
        The directions of rays are determined by ``forward`` and the rays
        with incorrect direction will be marked as invalid.

        :param BatchedRay ray: Incident rays.
        :param bool forward: Whether the incident rays propagate along positive-z direction.
        :return: Refracted rays with origin on this surface.
            A new :py:class:`~BatchedRay` object.
        :rtype: BatchedRay
        """
        magnitude = self._nt_threshold_strict / torch.finfo(ray.o.dtype).eps
        if torch.abs(ray.z - self.context.z).gt(magnitude).any():
            new_z = self.context.z - magnitude if forward else self.context.z + magnitude
            ray.march_to_(new_z, self.context.material_before.n(ray.wl, 'm'))

        if forward:
            valid = torch.logical_and(self._f(ray) >= 0, ray.d_z > 0)
        else:
            valid = torch.logical_and(self._f(ray) <= 0, ray.d_z < 0)
        ray.update_valid_(valid)

        return self.refract(self.intercept(ray), forward)

    def intercept(self, ray: BatchedRay) -> BatchedRay:
        """
        Returns a new :py:class:`~BatchedRay` whose directions are identical to those
        of ``ray`` and origins are the intersections of ``ray`` and this surface.
        The intersections are solved by `Newton's method
        <https://en.wikipedia.org/wiki/Newton's_method>`_ .
        The rays for which no intersection with sufficient precision, within the aperture
        and resulted from a positive marching distance will be marked as invalid.

        :param BatchedRay ray: Incident rays.
        :return: Intercepted rays.
        :rtype: BatchedRay
        """
        # TODO: optimize
        t = (self.context.z - ray.z) / ray.d_z
        cnt = 0  # equal to numbers of derivative computation
        new_ray = ray.clone(False)
        new_ray.norm_d_()
        with torch.no_grad():
            while True:
                new_ray.o = ray.o + new_ray.d_norm * t.unsqueeze(-1)  # do not compute opl for root finder
                f_value = self._f(new_ray)
                if torch.all(f_value.abs() < self._nt_threshold) or cnt >= self._nt_max_iteration:
                    break

                t = t - self._newton_descent(new_ray, f_value)
                cnt += 1

        # the second argument cannot be replaced by f_value because of computational graph
        t = t - self._newton_descent(new_ray, self._f(new_ray))
        ray = ray.march(t, self.context.material_before.n(ray.wl, 'm'))
        ray.update_valid_(
            self.aperture.pass_ray(ray) &
            (self._f(ray).abs() < self._nt_threshold_strict) &
            (t > 0)
        )
        return ray

    def normal(self, x: Ts, y: Ts) -> Ts:
        """
        Returns unit normal vector of the surface pointing to positive-z direction,
        i.e., from before the surface to behind it.

        :param Tensor x: x coordinate.
        :param Tensor y: y coordinate.
        :return: A tensor whose shape depends on ``x`` and ``y``, with an additional
            dimension of size 3 following.
        :rtype: Tensor
        """
        phpx, phpy = self.h_grad(x, y)
        f_grad = torch.stack((-phpx, -phpy, torch.ones_like(phpx)), dim=-1)
        return f_grad / f_grad.norm(2, -1, True)

    def refract(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        r"""
        Returns a new :py:class:`~BatchedRay` whose origins are identical to those
        of ``ray`` and directions are refracted by this surface.
        Refracted directions are computed by following equation:

        .. math::

            \mathbf{d}_2=\mu\mathbf{d}_1+\left(
                \sqrt{1-\mu^2\left(1-(\mathbf{n}\cdot\mathbf{d}_1)^2\right)}-
                \mu\mathbf{n}\cdot\mathbf{d}_1
            \right)\mathbf{n}

        where :math:`\mathbf{d}_1` and :math:`\mathbf{d}_2` are unit vectors of
        incident and refracted directions, :math:`\mu=n_1/n_2` is the ratio of
        refractive indices and :math:`\mathbf{n}` is unit normal vector of the surface,
        which is exactly :py:meth:`~normal`. Note that this equation applies only
        to forward ray tracing.
        The rays making the expression under the square root negative will be
        marked as invalid.

        :param BatchedRay ray: Incident rays.
        :param bool forward: Whether the incident rays propagate along positive-z direction.
        :return: Refracted rays with origin on this surface.
            A new :py:class:`~BatchedRay` object.
        :rtype: BatchedRay
        """
        ray = ray.clone(False)
        n = self.normal(ray.x, ray.y)
        i = ray.d_norm
        if forward:
            miu = self.context.material_before.n(ray.wl, 'm') / self.material.n(ray.wl, 'm')
        else:
            miu = self.material.n(ray.wl, 'm') / self.context.material_before.n(ray.wl, 'm')
            n = -n
        miu = miu.unsqueeze(-1)
        ni = torch.sum(n * i, dim=-1).unsqueeze(-1)
        nt2 = 1 - miu.square() * (1 - ni.square())
        t = torch.sqrt(nt2.relu_()) * n + miu * (i - ni * n)
        ray.d = t
        ray.update_valid_(nt2.squeeze(-1) > 0).norm_d_()
        return ray

    def _delegate(self) -> Ts:
        return self.distance

    def _f(self, ray: BatchedRay) -> Ts:
        return self.h_extended(ray.x, ray.y) + self.context.z - ray.z

    def _f_grad(self, ray: BatchedRay) -> Ts:
        phpx, phpy = self.h_grad_extended(ray.x, ray.y)
        return torch.stack((phpx, phpy, -torch.ones_like(phpx)), dim=-1)

    def _newton_descent(self, ray: BatchedRay, f_value: Ts) -> Ts:
        derivative_value = torch.sum(ray.d_norm * self._f_grad(ray), dim=-1)
        descent = f_value / (derivative_value + self._nt_epsilon)
        descent = torch.clip(descent, -self._nt_update_bound, self._nt_update_bound)
        return descent


class CircularSurface(Surface, metaclass=abc.ABCMeta):
    r"""
    Derived class of :py:class:`~Surface` for optical surfaces
    with circular symmetry, i.e. its property
    depends only on the radial distance :math:`r=\sqrt{x^2+y^2}`, in a group of lens.
    Therefore, their surface function can be written as
    :math:`h(x,y)=\hat{h}(x^2+y^2)=\hat{h}(r^2)`.
    Note that :math:`\hat{h}`
    takes as input squared radial distance for computational efficiency purpose.

     Despite the circular symmetry of the surface, its aperture is not necessarily
     circularly symmetric. In other words, ``aperture`` need not be an instance of
     :class:`CircularAperture`.

    :param float radius: Radius of effective region of the surface.
    :param material: Material following the surface. Either a :py:class:`~dnois.mt.Material`
        instance or a str representing the name of a registered material.
    :type material: :py:class:`~dnois.mt.Material` or str
    :param distance: Distance between the surface and the next one.
    :param Aperture aperture: Aperture of this surface. Default: a circular aperture whose
        radius is ``radius``.
    :param dict newton_config: Configuration for Newton's method.
        See :ref:`configuration_for_newtons_method` for details.
    """

    def __init__(
        self,
        radius: float,
        material: mt.Material | str,
        distance: Scalar,
        aperture: Aperture = None,
        newton_config: dict[str, Any] = None
    ):
        if aperture is None:
            aperture = CircularAperture(radius)
        super().__init__(material, distance, aperture, newton_config)
        self.radius: float = radius  #: Radius of effective region of the surface.

    @abc.abstractmethod
    def h_r2(self, r2: Ts) -> Ts:
        r"""
        Computes surface function :math:`\hat{h}(r^2)`.

        :param Tensor r2: Squared radial distance.
        :return: Corresponding value of the surface function.
        :rtype: Tensor
        """
        pass

    @abc.abstractmethod
    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        r"""
        Computes the partial derivatives of surface function
        :math:`\pfrac{h(x,y)}{x}` and :math:`\pfrac{h(x,y)}{y}`.

        :param Tensor x: x coordinate.
        :param Tensor y: y coordinate.
        :param Tensor r2: Squared radial distance. It can be passed in to avoid
            repeated computation if already computed outside this method.
        :return: Corresponding value of two partial derivatives.
        :rtype: tuple[Tensor, Tensor]
        """
        pass

    def h(self, x: Ts, y: Ts) -> Ts:
        return self.h_r2(x.square() + y.square())

    def h_extended(self, x: Ts, y: Ts) -> Ts:
        return self.h_extended_r2(x.square() + y.square())

    def h_grad_extended(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        r"""
        Computes partial derivatives of extended surface function:
        :math:`\pfrac{\tilde{h}(x,y)}{x}` and :math:`\pfrac{\tilde{h}(x,y)}{y}`.
        See :py:meth:`~h_extended`.

        :param Tensor x: x coordinate.
        :param Tensor y: y coordinate.
        :param Tensor r2: Squared radial distance. It can be passed in to avoid
            repeated computation if already computed outside this method.
        :return: Corresponding value of two partial derivatives.
        :rtype: tuple[Tensor, Tensor]
        """
        if r2 is None:
            r2 = x.square() + y.square()
        lim2 = self.geo_radius.square()
        phpx, phpy = self.h_grad(x, y, r2)
        mask = r2 <= lim2
        return torch.where(mask, phpx, 0), torch.where(mask, phpy, r2)

    def h_extended_r2(self, r2: Ts) -> Ts:
        r"""
        Computes extended version of :math:`\hat{h}(r^2)`.
        See :py:meth:`~h_r2` and :py:meth:`~h_extended`.
        """
        lim2 = self.geo_radius.square()
        return torch.where(r2 <= lim2, self.h_r2(r2), self.h_r2(lim2))

    @property
    def geo_radius(self) -> Ts:
        """
        Geometric radius of the surface, i.e. maximum radial distance that makes
        the surface function mathematically meaningful. A 0D tensor.

        :type: Tensor
        """
        return self.new_tensor(self.radius)

    def _f(self, ray: BatchedRay) -> Ts:
        return self.h_extended_r2(ray.r2) + self.context.z - ray.z

    def _f_grad(self, ray: BatchedRay) -> Ts:
        phpx, phpy = self.h_grad_extended(ray.x, ray.y, ray.r2)
        return torch.stack((phpx, phpy, -torch.ones_like(phpx)), dim=-1)

    def _newton_descent(self, ray: BatchedRay, f_value: Ts) -> Ts:
        derivative_value = torch.sum(ray.d_norm * self._f_grad(ray), dim=-1)
        descent = f_value / (derivative_value + self._nt_epsilon)
        descent = torch.clip(descent, -self._nt_update_bound, self._nt_update_bound)
        return descent


class SurfaceList(nn.ModuleList, collections.abc.MutableSequence):
    """
    A sequential container of surfaces. This class is derived from
    :py:class:`torch.nn.ModuleList` and implements
    :py:class:`collections.abc.MutableSequence` interface.
    So its instance can be regarded as both a PyTorch module
    and a list of :py:class:`Surface`.

    :param surfaces: A sequence of :py:class:`Surface` objects. Default: ``[]``.
    :type surfaces: Sequence[Surface]
    :param env_material: The material before the first surface.
    :type env_material: :py:class:`~dnois.mt.Material`
    """
    _force_surface: bool = True
    __call__: Callable[..., BatchedRay]  # for return type hint in IDE

    def __init__(
        self,
        surfaces: Sequence[Surface] = None,
        env_material: mt.Material | str = 'vacuum',
    ):
        super().__init__()
        if surfaces is None:
            surfaces = []
        if not isinstance(env_material, mt.Material):
            env_material = mt.get(env_material)

        # This is needed to facilitate MutableSequence operations
        # because torch.nn.ModuleList saves submodules like a dict rather than list
        self._slist: list[Surface] = []
        self._stop_idx = None
        #: Material before the first surface.
        self.env_material: mt.Material = env_material

        self.extend(surfaces)

    def __contains__(self, item) -> bool:
        """:meta private:"""
        return self._slist.__contains__(item)

    def __delitem__(self, key):
        """:meta private:"""
        super().__delitem__(key)
        self._slist.__getitem__(key).context = None
        self._slist.__delitem__(key)

    def __getitem__(self, item):
        """:meta private:"""
        return self._slist.__getitem__(item)

    def __iadd__(self, other: Sequence[Surface]) -> Self:
        """:meta private:"""
        self.extend(other)
        return self

    def __iter__(self):
        """:meta private:"""
        return self._slist.__iter__()

    def __len__(self):
        """:meta private:"""
        return self._slist.__len__()

    def __reversed__(self) -> 'SurfaceList':
        """:meta private:"""
        return SurfaceList(list(self._slist.__reversed__()), self.env_material)

    def __setitem__(self, key: int, value: Surface):
        """:meta private:"""
        self._welcome(value)
        super().__setitem__(key, value)
        self._slist.__setitem__(key, value)

    def __add__(self, other: Sequence[Surface]) -> 'SurfaceList':
        """:meta private:"""
        return SurfaceList(self._slist + list(other), self.env_material)

    def __repr__(self) -> str:
        """:meta private:"""
        _repr = super().__repr__()[:-1]  # remove that last parentheses
        _repr += f'  env_material={repr(self.env_material)}\n)'
        return _repr

    def __dir__(self):
        """:meta private:"""
        return super().__dir__() + ['env_material']

    def append(self, surface: Surface):
        """:meta private:"""
        self._welcome(surface)
        super().append(surface)

    def clear(self):
        """:meta private:"""
        for s in self._slist:
            s.context = None
        self._slist.clear()
        self._super_clear()

    def count(self, value: Surface) -> int:
        """:meta private:"""
        return self._slist.count(value)

    def extend(self, surfaces: Sequence[Surface]):
        """:meta private:"""
        self._welcome(*surfaces)
        super().extend(surfaces)

    def index(self, value: Surface, start: int = 0, stop: int = ...) -> int:
        """:meta private:"""
        if stop is ...:
            return self._slist.index(value, start)
        else:
            return self._slist.index(value, start, stop)

    def insert(self, index: int, surface: Surface):
        """:meta private:"""
        self._welcome(surface)
        super().insert(index, surface)
        self._slist.insert(index, surface)

    def pop(self, index: int = -1) -> Surface:
        """:meta private:"""
        s = super().pop(index)
        s.context = None
        return s

    def remove(self, value: Surface):
        """:meta private:"""
        idx = self.index(value)
        self.pop(idx)

    def reverse(self):
        """:meta private:"""
        ss = self._slist.copy()
        self._slist.clear()
        self._super_clear()
        self.extend(ss)

    def add_module(self, name: str, module: nn.Module):
        """:meta private:"""
        if name.isdigit() and isinstance(module, Surface):
            self._slist.insert(int(name), module)
        super().add_module(name, module)

    def forward(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        """
        Traces rays incident on the first surface and returns rays
        passing the last surface, or reversely if ``forward`` is ``False``.

        :param BatchedRay ray: Input rays.
        :param bool forward: Whether rays are forward or not.
        :return: Output rays.
        :rtype: BatchedRay
        """
        for s in (self._slist if forward else reversed(self._slist)):
            try:
                ray = s(ray, forward)
            except Exception as e:
                idx = self.index(s)
                e.add_note(f'This exception is raised during the forward pass of surface {idx}')
                raise e
        return ray

    @property
    def surfaces(self) -> list[Surface]:
        """
        Returns a list of contained surfaces.

        :type: list[Surface]
        """
        return [s for s in self._slist]

    @property
    def length(self) -> Ts | None:
        """
        Returns the distance between baselines of the first and the last surfaces
        as a 0D tensor. If there is no more than one surface, returns ``None``.

        :type: Tensor or None
        """
        if len(self._slist) < 2:
            return None
        return sum(s.distance for s in self._slist[:-1])

    @property
    def total_length(self) -> Ts | None:
        """
        Returns the sum of :py:attr:`Surface.distance` of all the surfaces
        as a 0D tensor. If there is no surface, returns ``None``.

        :type: Tensor or None
        """
        if len(self._slist) == 0:
            return None
        return sum(s.distance for s in self._slist)

    @property
    def stop_idx(self) -> int:
        """
        Index of the aperture stop. Returns ``None`` if no stop is found.

        :type: int
        """
        return self._stop_idx

    @property
    def stop(self) -> Surface:
        """
        The aperture stop object. Returns ``None`` if no stop is found.
        Note that it need not return an instance of :py:class:`CircularStop`.

        :type: :py:class:`CircularStop`
        """
        idx = self._stop_idx
        return None if idx is None else self._slist[idx]

    def _welcome(self, *new: Surface):
        for surface in new:
            if self._force_surface and not isinstance(surface, Surface):
                msg = f'An instance of {Surface.__name__} expected, got {type(surface)}'
                raise TypeError(msg)
            if surface.context is not None:
                raise RuntimeError(f'A surface already contained in a lens group '
                                   f'cannot be inserted to another one')
            surface.context = Context(surface, self)

        for s1 in self._slist:
            for s2 in new:
                if id(s1) == id(s2):
                    raise ValueError('Trying to add a surface into a lens group containing it')

    def _super_clear(self):
        for idx in range(len(self) - 1, -1, -1):
            super().__delitem__(idx)
