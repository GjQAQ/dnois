"""
This module provides a series of classes representing various optical materials
with several different dispersion formula, such as Cauchy formula, Schott formula
and Sellmeier formula, etc.

All the material classes are derived from :py:class:`Material` and share its
constructor arguments. They have also a method :py:meth:`~Material.n`
to compute the refractive index for given wavelength. Each class implements
this method by its own dispersion formula. The main reference for these dispersion
types is Zemax.

This module maintains a material library to add, delete or retrieve materials.
See :ref:`accessing_materials`.
"""

import abc
import json
from pathlib import Path
import re

import torch

from . import utils, base, conf, torch as _t
from .base.typing import Numeric, Union, Any, Self, cast

__all__ = [
    'air',
    'vacuum',

    'dispersion_types',
    'get',
    'list_all',
    'load',
    'refractive_index',
    'register',
    'registered',
    'remove',
    'save',

    'Cauchy',
    'Conrady',
    'Constant',
    'Herzberger',
    'Material',
    'Schott',
    'Sellmeier1',
    'Sellmeier2',
    'Sellmeier3',
    'Sellmeier4',
    'Sellmeier5',

    'MaterialNotFoundError',
]


def _format_flist(flist: list[float]) -> str:
    return f'[{", ".join(f"{utils.fmt(c)}" for c in flist)}]'


# TODO: replace deep copy of registered materials with shallow copy
class Material(base.AsJsonMixIn, metaclass=abc.ABCMeta):
    """
    Class representing an optical material type.

    :param str name: Name of the material.
    :param float min_wl: Minimum applicable wavelength in ``default_unit``. Default: 0.
    :param float max_wl: Maximum applicable wavelength in ``default_unit``. Default: infinity.
    :param float ref_t: Reference temperature in degree Celsius. Default: 20.
    """
    __slots__ = ('name', 'min_wl', 'max_wl', 'ref_t', 'thermal_d', 'thermal_e', 'ltk')
    _forbidden_name = ['', 'None', 'none', 'null']

    def __init__(
        self,
        name: str,
        min_wl: float = None,
        max_wl: float = None,
        ref_t: float = 20,
        d0: float = 0.,
        d1: float = 0.,
        d2: float = 0.,
        e0: float = 0.,
        e1: float = 0.,
        ltk: float = 0.,
    ):
        if name in self._forbidden_name:
            raise ValueError(f'Material name cannot be {name}')
        if min_wl is not None and min_wl < 0 or max_wl is not None and max_wl < 0:
            raise ValueError(f'Limits of wavelength cannot be negative, but got {min_wl} and {max_wl}')
        if min_wl is not None and max_wl is not None and max_wl < min_wl:
            raise ValueError(f'Maximum wavelength ({max_wl}) cannot be smaller than minimum wavelength ({min_wl})')

        #: Name of the material.
        self.name = name
        #: Minimum wavelength valid for the material.
        self.min_wl = min_wl if min_wl is not None else 0.
        #: Maximum wavelength valid for the material.
        self.max_wl = max_wl if max_wl is not None else float('inf')
        #: Reference temperature in degree Celsius.
        self.ref_t = ref_t
        self.thermal_d = (d0, d1, d2)
        self.thermal_e = (e0, e1)
        self.ltk = ltk

    def __repr__(self):
        return f'{self.__class__.__name__}({self._repr()})'

    def __str__(self):
        return self.name

    def n(self, wl: Numeric, t: float = None, p: float = None, relative: bool = True) -> Numeric:
        """
        Computes refractive index.

        .. note::
            The result is not dependent on ``t`` if :data:`~dnois.conf.temperature_affect_n`
            is ``False``, similarly for ``p`` and :data:`~dnois.conf.pressure_affect_n`.

        :param wl: Wavelength measured in air under given condition.
        :type wl: float or Tensor
        :param float t: Temperature in degree Celsius. Default: :data:`~dnois.conf.default_temperature`.
        :param float p: Pressure in atm. Default: :data:`~dnois.conf.default_pressure`.
        :param bool relative: Whether to return relative refractive index or the
            absolute one otherwise. Default: ``True``.
        :return: Refractive index.
        """
        if relative:
            return self.n_rel(wl, t, p)
        else:
            return self.n_abs(wl, t, p)

    def n_rel(self, wl: Numeric, t: float = None, p: float = None) -> Numeric:
        """
        Computes refractive index relative to :class:`Air`.

        See :meth:`.n` for description of parameters.
        """
        wl = self._make_wl(wl)
        if not conf.temperature_affect_n and not conf.pressure_affect_n:
            return self._dispersion_formula(wl)

        n_abs, n_air = self._n_impl(wl, p, t)
        n_rel = n_abs / n_air  # relative n measured in given condition
        return n_rel

    def n_abs(self, wl: Numeric, t: float = None, p: float = None) -> Numeric:
        """
        Computes absolute refractive index.

        See :meth:`.n` for description of parameters.
        """
        wl = self._make_wl(wl)
        if not conf.temperature_affect_n and not conf.pressure_affect_n:
            return self._dispersion_formula(wl) * _air_n(wl * wl)

        n_abs, _ = self._n_impl(wl, p, t)
        return n_abs

    @abc.abstractmethod
    def _dispersion_formula(self, wl: Numeric) -> Numeric:
        pass

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        return {
            'type': self.__class__.__name__,
            'name': self.name,
            'min_wl': self.min_wl,
            'max_wl': self.max_wl,
            'ref_t': self.ref_t,
            'thermal_d': self.thermal_d,
            'thermal_e': self.thermal_e,
            'ltk': self.ltk,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        if cls is not Material:
            d.pop('type', None)
            return cls(**d)  # default implementation of eponymous method

        mt_type = d['type']
        subs = utils.subclasses(cls)
        for sub in subs:
            if sub.__name__ == mt_type:
                return cast(type[Material], sub).from_dict(d)  # calling eponymous method of subclass
        raise RuntimeError(utils.invalid_option_msg('material type', mt_type, dispersion_types(True)))

    def _n_impl(self, wl, p, t):
        if t is None:
            t = conf.default_temperature
        if p is None:
            p = conf.default_pressure
        wl2 = wl * wl
        n_air = _air_n(wl2, t, p)  # n of air in given condition
        n_air_ref = _air_n(wl2, self.ref_t, 1.)  # n of air in reference condition
        wl_ref = wl * n_air / n_air_ref  # wavelength measured in reference condition
        n_ref_rel = self._dispersion_formula(wl_ref)  # relative n measured in reference condition
        n_ref_abs = n_ref_rel * n_air_ref  # absolute n measured in reference condition
        dt = t - self.ref_t
        dn = _t.polynomial(dt, self.thermal_d) * dt
        dn = dn + _t.polynomial(dt, self.thermal_e) * dt / (wl2 - self.ltk * abs(self.ltk))
        dn = dn * (n_ref_rel * n_ref_rel - 1) / (2 * n_ref_rel)
        n_abs = n_ref_abs + dn  # absolute n measured in given condition
        return n_abs, n_air

    def _repr(self) -> str:
        return ', '.join([
            f'name={self.name}',
            f'domain=({utils.fmt(self.min_wl)}um, {utils.fmt(self.max_wl)}um)',
            f'T={utils.fmt(self.ref_t)}°C',
            f'D0={utils.fmt(self.thermal_d[0])}',
            f'D1={utils.fmt(self.thermal_d[1])}',
            f'D2={utils.fmt(self.thermal_d[2])}',
            f'E0={utils.fmt(self.thermal_e[0])}',
            f'E1={utils.fmt(self.thermal_e[1])}',
            f'Ltk={utils.fmt(self.ltk)}',
        ])

    def _make_wl(self, wl: Numeric) -> Numeric:
        wl = base.Length.default_to(wl, 'um')
        m1, m2 = (wl.min().item(), wl.max().item()) if torch.is_tensor(wl) else (wl, wl)
        if m1 < self.min_wl * (1 - conf.detection_wl_eps) or m2 > self.max_wl * (1 + conf.detection_wl_eps):
            raise ValueError(f'Unsupported wavelength for material "{self.name}": {wl}um')
        else:
            return wl


class Constant(Material):
    """
    Material with constant optical properties.

    :param float n: Constant refractive index.

    See :class:`Material` for descriptions for other parameters.
    """
    __slots__ = ('refractive_index',)

    def __init__(self, name: str, n: float, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.refractive_index: float = n  #: Refractive index.

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['n'] = self.refractive_index
        return d

    def _repr(self) -> str:
        return super()._repr() + f', n={utils.fmt(self.refractive_index)}'

    def _dispersion_formula(self, wl: Numeric) -> Numeric:
        n = self.refractive_index
        return torch.full_like(wl, n) if torch.is_tensor(wl) else n


class Air(Material):
    r"""
    Air in normal temperature and pressure, whose dispersion formula is:

    .. math::
        n=1+\left(6432.8+\frac{2949810}{146\lambda^2-1}+\frac{25540}{41\lambda^2-1}\right)10^{-8}

    See :class:`Material` for descriptions of parameters.
    """

    def n_rel(self, wl: Numeric, t: float = None, p: float = None) -> Numeric:
        return torch.ones_like(wl) if torch.is_tensor(wl) else 1.

    def n_abs(self, wl: Numeric, t: float = None, p: float = None) -> Numeric:
        if t is None:
            t = conf.default_temperature
        if p is None:
            p = conf.default_pressure

        wl = self._make_wl(wl)
        return _air_n(wl * wl, t, p)

    def _dispersion_formula(self, wavelength: Numeric) -> Numeric:
        return torch.ones_like(wavelength) if torch.is_tensor(wavelength) else 1


class Cauchy(Material):
    r"""
    Material for which Cauchy formula:

    .. math::
        n=A+\frac{B}{\lambda^2}+\frac{C}{\lambda^4}

    :param float a: :math:`A` in Cauchy formula.
    :param float b: :math:`B` in Cauchy formula.
    :param float c: :math:`C` in Cauchy formula.

    See :class:`Material` for descriptions for other parameters.
    """
    __slots__ = ('a', 'b', 'c')

    def __init__(self, name: str, a: float, b: float, c: float, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.a = a  #: :math:`A` in Cauchy formula.
        self.b = b  #: :math:`B` in Cauchy formula.
        self.c = c  #: :math:`C` in Cauchy formula.

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['a'] = self.a
        d['b'] = self.b
        d['c'] = self.c
        return d

    def _dispersion_formula(self, wl: Numeric) -> Numeric:
        iw2 = 1 / cast(Numeric, wl ** 2)
        n = _t.polynomial(iw2, (self.a, self.b, self.c))
        return n

    def _repr(self) -> str:
        return super()._repr() + f', A={utils.fmt(self.a)}, B={utils.fmt(self.b)}, C={utils.fmt(self.c)}'


class Schott(Material):
    r"""
    Materials described by Schott formula:

    .. math::
        n^2=a_0+a_1\lambda^2+a_2\lambda^{-2}+a_3\lambda^{-4}+a_4\lambda^{-6}+a_5\lambda^{-8}

    :param list[float] coefficients: The six coefficients in Schott formula.

    See :class:`Material` for descriptions for other parameters.
    """
    __slots__ = ('coefficients',)

    def __init__(self, name: str, coefficients: list[float], *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        if len(coefficients) != 6:
            raise ValueError(f'Number of coefficients in Schott formula must be 6.')
        self.coefficients = coefficients  #: The six coefficients in Schott formula.

    def __getattr__(self, name: str):
        if len(name) == 2 and name[0] == 'a' and name[1].isdigit():
            return self.coefficients[int(name[1]) - 1]
        return super().__getattribute__(name)

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['coefficients'] = self.coefficients
        return d

    def _dispersion_formula(self, wl: Numeric) -> Numeric:
        wl2 = wl * wl
        iw2 = 1 / wl2
        iw4 = iw2 * iw2
        a = self.coefficients
        n2 = a[0] + a[1] * wl2 + a[2] * iw2 + a[3] * iw4 + a[4] * (iw4 * iw2) + a[5] * (iw4 * iw4)
        n = n2 ** 0.5
        return n

    def _repr(self) -> str:
        return f'{super()._repr()}, coefficients={_format_flist(self.coefficients)}'


class _Sellmeier(Material):
    __slots__ = ('ks', 'ls')
    _n_terms: int

    def __init__(self, name: str, ks: list[float], ls: list[float], *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        if len(ks) != self._n_terms or len(ls) != self._n_terms:
            raise ValueError(f'Numbers of K and L coefficients should be {self._n_terms}.')
        self.ks = ks  #: The coefficients :math:`K_i` s in Sellmeier{num} formula.
        self.ls = ls  #: The coefficients :math:`L_i` s in Sellmeier{num} formula.

    def __getattr__(self, name: str):
        if re.match(r'^[kl][1-9]$', name):
            return getattr(self, f'{name[0]}s')[int(name[1]) - 1]
        return super().__getattribute__(name)

    def __setattr__(self, key, value):
        if re.match(r'^[kl][1-9]$', key):
            if key[0] == 'k':
                self.ks[int(key[1]) - 1] = value
            else:
                self.ls[int(key[1]) - 1] = value
        else:
            super().__setattr__(key, value)

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['ks'] = self.ks
        d['ls'] = self.ls
        return d

    def _dispersion_formula(self, wl: Numeric) -> Numeric:
        w2 = cast(Numeric, wl ** 2)
        n2 = 1 + sum([kc * w2 / (w2 - lc) for kc, lc in zip(self.ks, self.ls)])
        n = n2 ** 0.5
        return n

    def _repr(self) -> str:
        return f'{super()._repr()}, K={_format_flist(self.ks)}, L={_format_flist(self.ls)}'


def _make_sellmeier(num: int, n_terms: int) -> type[_Sellmeier]:
    cls = type(f'Sellmeier{num}', (_Sellmeier,), {'_n_terms': n_terms})
    cls.__doc__ = fr"""
    Materials described by Sellmeier{num} formula:

    .. math::
        n^2-1=\sum_{{i=1}}^{n_terms}\frac{{K_i\lambda^2}}{{\lambda^2-L_i}}
        
    :param list[float] ks: The coefficients :math:`K_i,i=1,\cdots,{n_terms}` in Sellmeier{num} formula.
    :param list[float] ls: The coefficients :math:`L_i,i=1,\cdots,{n_terms}` in Sellmeier{num} formula.
    
    See :class:`Material` for descriptions for other parameters.
    """
    return cast(type[_Sellmeier], cls)


Sellmeier1 = _make_sellmeier(1, 3)
Sellmeier3 = _make_sellmeier(3, 4)
Sellmeier5 = _make_sellmeier(5, 5)


class Sellmeier2(Material):
    r"""
    Materials described by Sellmeier2 formula:

    .. math::
        n^2-1=A+\frac{B_1\lambda^2}{\lambda^2-\lambda_1^2}+\frac{B_2}{\lambda^2-\lambda_2^2}

    :param float a: :math:`A` in Sellmeier2 formula.
    :param float b1: :math:`B_1` in Sellmeier2 formula.
    :param float b2: :math:`B_2` in Sellmeier2 formula.
    :param float wl1: :math:`\lambda_1` in Sellmeier2 formula.
    :param float wl2: :math:`\lambda_2` in Sellmeier2 formula.

    See :class:`Material` for descriptions for other parameters.
    """
    __slots__ = ('a_pp', 'b1', 'b2', 'swl1', 'swl2')

    def __init__(self, name: str, a: float, b1: float, b2: float, wl1: float, wl2: float, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.a_pp = a + 1  #: :math:`A+1` in Sellmeier2 formula.
        self.b1 = b1  #: :math:`B_1` in Sellmeier2 formula.
        self.b2 = b2  #: :math:`B_2` in Sellmeier2 formula.
        self.swl1 = wl1 * wl1  #: :math:`\lambda_1^2` in Sellmeier2 formula.
        self.swl2 = wl2 * wl2  #: :math:`\lambda_2^2` in Sellmeier2 formula.

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['a'] = self.a_pp - 1
        d['b1'] = self.b1
        d['b2'] = self.b2
        d['wl1'] = self.swl1 ** 0.5
        d['wl2'] = self.swl2 ** 0.5
        return d

    def _dispersion_formula(self, wl: Numeric) -> Numeric:
        w2 = cast(Numeric, wl ** 2)
        n2 = self.a_pp + self.b1 * w2 / (w2 - self.swl1) + self.b2 / (w2 - self.swl2)
        n = n2 ** 0.5
        return n

    def _repr(self) -> str:
        return (f'{super()._repr()}, '
                f'A={utils.fmt(self.a_pp - 1)}, '
                f'B1={utils.fmt(self.b1)}, B2={utils.fmt(self.b2)}, '
                f'Wl1={utils.fmt(self.swl1 ** 0.5)}, Wl2={utils.fmt(self.swl2 ** 0.5)}')


class Sellmeier4(Material):
    r"""
    Materials described by Sellmeier4 formula:

    .. math::
        n^2=A+\frac{B\lambda^2}{\lambda^2-C}+\frac{D\lambda^2}{\lambda^2-E}

    :param float a: :math:`A` in Sellmeier4 formula.
    :param float b: :math:`B` in Sellmeier4 formula.
    :param float c: :math:`C` in Sellmeier4 formula.
    :param float d: :math:`D` in Sellmeier4 formula.
    :param float e: :math:`E` in Sellmeier4 formula.

    See :class:`Material` for descriptions for other parameters.
    """
    __slots__ = ('a', 'b', 'c', 'd', 'e')

    def __init__(self, name: str, a: float, b: float, c: float, d: float, e: float, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.a = a  #: :math:`A` in Sellmeier4 formula.
        self.b = b  #: :math:`B` in Sellmeier4 formula.
        self.c = c  #: :math:`C` in Sellmeier4 formula.
        self.d = d  #: :math:`D` in Sellmeier4 formula.
        self.e = e  #: :math:`E` in Sellmeier4 formula.

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        for k in ['a', 'b', 'c', 'd', 'e']:
            d[k] = getattr(self, k)
        return d

    def _dispersion_formula(self, wl: Numeric) -> Numeric:
        w2 = cast(Numeric, wl ** 2)
        n2 = self.a + self.b * w2 / (w2 - self.c) + self.d * w2 / (w2 - self.e)
        n = n2 ** 0.5
        return n

    def _repr(self) -> str:
        return (f'{super()._repr()}, '
                f'A={utils.fmt(self.a)}, '
                f'B={utils.fmt(self.b)}, '
                f'C={utils.fmt(self.c)}, '
                f'D={utils.fmt(self.d)}, '
                f'E={utils.fmt(self.e)}')


class Herzberger(Material):
    r"""
    Materials described by Herzberger formula:

    .. math::
        n=A+BL+CL^2+D\lambda^2+E\lambda^4+F\lambda^6,\\
        L=\frac{1}{\lambda^2-0.028}

    :param list[float] coefficients: The six coefficients in Herzberger formula.

    See :class:`Material` for descriptions for other parameters.
    """
    __slots__ = ('coefficients',)

    def __init__(self, name: str, coefficients: list[float], *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        if len(coefficients) != 6:
            raise ValueError(f'Number of coefficients in Herzberger formula must be 6.')
        self.coefficients = coefficients  #: The six coefficients in Herzberger formula.

    def __getattr__(self, name: str):
        if len(name) == 2 and name[0] == 'a' and name[1].isdigit():
            return self.coefficients[int(name[1]) - 1]
        return super().__getattribute__(name)

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['coefficients'] = self.coefficients
        return d

    def _dispersion_formula(self, wl: Numeric) -> Numeric:
        w2 = cast(Numeric, wl ** 2)
        m = 1 / (w2 - 0.028)
        _1, _2, _3, _4, _5, _6 = self.coefficients
        n = _1 + m * (_2 + m * _3) + w2 * (_4 + w2 * (_5 + w2 * _6))
        return n

    def _repr(self) -> str:
        return f'{super()._repr()}, coefficients={_format_flist(self.coefficients)}'


class Conrady(Material):
    r"""
    Materials described by Conrady formula:

    .. math::
        n=n_0+\frac{A}{\lambda}+\frac{B}{\lambda^{3.5}}

    :param float n0: :math:`n_0` in Conrady formula.
    :param float a: :math:`A` in Conrady formula.
    :param float b: :math:`B` in Conrady formula.

    See :class:`Material` for descriptions for other parameters.
    """
    __slots__ = ('n0', 'a', 'b')

    def __init__(self, name: str, n0: float, a: float, b: float, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.n0 = n0  #: :math:`n_0` in Conrady formula.
        self.a = a  #: :math:`a` in Conrady formula.
        self.b = b  #: :math:`b` in Conrady formula.

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['n0'] = self.n0
        d['a'] = self.a
        d['b'] = self.b
        return d

    def _dispersion_formula(self, wl: Numeric) -> Numeric:
        n = self.n0 + self.a / wl + self.b / cast(Numeric, wl ** 3.5)
        return n

    def _repr(self) -> str:
        return f'{super()._repr()}, n0={utils.fmt(self.n0)}, A={utils.fmt(self.a)}, B={utils.fmt(self.b)}'


def _ref_n(wl2):
    return 1 + 6.4328e-5 + 2.94981e-2 * wl2 / (146 * wl2 - 1) + 2.5540e-4 * wl2 / (41 * wl2 - 1)


def _air_n(wl2, t=None, p=None):  # p default to 1, t default to 15
    n = _ref_n(wl2)
    if not conf.temperature_affect_n and not conf.pressure_affect_n:
        return n
    if p is None and t is None:
        return n

    numerator = n - 1
    if conf.pressure_affect_n and p is not None:
        numerator = numerator * p
    if not conf.temperature_affect_n or t is None:
        return 1 + numerator
    else:
        return 1 + numerator / (1 + (t - 15) * 3.4785e-3)


class MaterialNotFoundError(LookupError):
    """Raised when a material is not found in the database."""

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(f'Material {name} not found', *args, **kwargs)
        self.material_name = name


def _resolve_name(name):
    if ':' in name:
        qualifier, name = name.split(':')
    else:
        qualifier = None
    return name, qualifier


def get(name: str, default_none: bool = False) -> Union[Material, None]:
    """
    Get material by name from material library.

    :param str name: Name of the material.
    :param bool default_none: If true, return ``None`` when the material does not exist.
        Otherwise, an error is raised.
    :return: Specified material.
    :rtype: Material
    :raises MaterialNotFoundError: If the material does not exist and ``default_none`` is ``False``.
    """
    name, qualifier = _resolve_name(name)

    m = _lib.get(name, None)
    if m is None or len(m) == 0:
        if default_none:
            return None
        raise MaterialNotFoundError(name)

    if not qualifier:
        if '' in m:
            return m['']  # empty qualifier is default
        m = next(iter(m.values()))  # pick an arbitrary material
    else:
        m = m.get(qualifier, None)
    if m is None:
        if default_none:
            return None
        raise MaterialNotFoundError(f'{qualifier}:{name}')
    return m


def registered(name: str) -> bool:
    """
    Check if a material is registered in material library by name.

    .. warning::
        Materials with the same name are seen as identical by this function.

    :param str name: Name of the material.
    :return: If the material is registered.
    :rtype: bool
    """
    return get(name, True) is not None


def search(pattern: str | re.Pattern) -> list[Material]:
    """
    Search materials by regular expression.

    :param str or re.Pattern pattern: The regular expression pattern.
    :return: List of materials whose names match the pattern.
    :rtype: list[Material]
    """
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    return [v for k, v in lib() if pattern.search(k)]


def register(material: Material, exist_ok: bool = False):
    """
    Add a new class of material into material library.

    :param Material material: The material instance.
    :param bool exist_ok: If ``False``, raise :py:exc:`KeyError` if the material already exists.
        Otherwise, overwrite the existing material. Default: ``False``.
    """
    name = material.name
    if registered(name) and not exist_ok:
        raise KeyError(f'Material {name} already exists.')

    name, qualifier = _resolve_name(name)
    if name not in _lib:
        _lib[name] = {}
    if qualifier is None:
        qualifier = ''
    _lib[name][qualifier] = material


def refractive_index(wl: Numeric, material: str, t: float = None, p: float = None) -> Numeric:
    """
    Compute refractive index for given wavelength and material.

    :param wl: Specified wavelength.
    :type: float or Tensor
    :param str material: Specified material.
    :return: Refractive index.
    :rtype: float or Tensor
    """
    m = get(material)
    n = m.n(wl, t, p)
    return n


def list_all() -> list[str]:
    """
    List all available materials in material library.

    :return: List of the names of available materials.
    :rtype: list[str]
    """
    keys = []
    for name, d in _lib.items():
        for qualifier, m in d.items():
            keys.append(f'{qualifier}:{name}' if qualifier else name)
    return keys


def lib() -> dict[str, Material]:
    """
    Return a snapshot of the material library.

    :return: A map from qualified material name to material instance.
    :rtype: dict[str, Material]
    """
    snapshot = {}
    for name, d in _lib.items():
        for qualifier, m in d.items():
            snapshot[f'{qualifier}:{name}'] = m
    return snapshot


def remove(name: str, ignore_if_absent: bool = False):
    """
    Remove a material from material library.

    .. warning::
        If no qualifier in ``name``, all materials with the same name will be removed.
        Specify an empty qualifier if only the material with empty qualifier should be removed.

    :param str name: Name of the material.
    :param bool ignore_if_absent: If true, ignore material if it does not exist,
        or raise an error if false.
    :raises MaterialNotFoundError: If the material does not exist and ``ignore_if_absent`` is ``False``.
    """
    if not registered(name):
        if ignore_if_absent:
            return
        raise MaterialNotFoundError(name)
    name, qualifier = _resolve_name(name)
    if qualifier is None:
        del _lib[name]
    else:
        del _lib[name][qualifier]
        if len(_lib[name]) == 0:
            del _lib[name]


def update(name: str, material: Material):
    """
    Update a registered material.

    :param str name: Original name of the material.
    :param Material material: The new material instance.
    """
    name, qualifier = _resolve_name(name)
    if qualifier is None:
        qualifier = ''

    if name not in _lib:
        _lib[name] = {}
    _lib[name][qualifier] = material


def dispersion_types(name_only: bool = False) -> list[type[Material]] | list[str]:
    """
    Returns a list of accessible subclasses of :class:`Material` in lexicographic order.
    This can be used to recognize material types supported by dnois.

    :param bool name_only: If ``True``, returns class names, otherwise returns class objects.
    :return: A list of subclasses of :class:`.Material`.
    :rtype: list[type[Material]] or list[str]
    """
    sub_list = utils.subclasses(Material)
    if name_only:
        return [sub.__name__ for sub in sub_list]
    else:
        return cast(list, sub_list)


def save(file):
    """
    Save the material library to a JSON file.

    :param file: The JSON file to save. Either its path (``str`` or ``pathlib.Path``)
        or a file-like object.
    """
    materials = [m.to_dict() for m in lib().values()]
    json.dump(materials, file, separators=(',', ':'))


def load(file, exist_ok: bool = False):
    """
    Load the material library from a JSON file.

    :param file: The JSON file to load. Either its path (``str`` or ``pathlib.Path``)
        or a file-like object.
    :param bool exist_ok: If ``False``, raise :exc:`KeyError` if the material already exists.
        Otherwise, overwrite the existing material. Default: ``False``.
    """
    if isinstance(file, str):
        file = Path(file)
    if isinstance(file, Path):
        with file.open('r', encoding='utf-8') as fp:
            materials = json.load(fp)
    else:
        materials = json.load(file)

    for m in materials:
        register(Material.from_dict(m), exist_ok=exist_ok)


air: Air = Air('air')
vacuum: Constant = Constant('vacuum', 1.)
_lib: dict[str, dict[str, Material]] = {
    'air': {'': air},
    'vacuum': {'': vacuum},
}
