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
import importlib.resources
import json
from pathlib import Path
import re

import torch

from . import utils, base
from .base.typing import Numeric, Union, Any, Self, cast

__all__ = [
    'air',
    'vacuum',

    'dispersion_types',
    'get',
    'is_available',
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
]

RANGE_CHECK_EPS = 1e-5


def _format_flist(flist: list[float]) -> str:
    return f'[{", ".join(f"{utils.fmt(c)}" for c in flist)}]'


# TODO: replace deep copy of registered materials with shallow copy
class Material(base.AsJsonMixIn, metaclass=abc.ABCMeta):
    """
    Class representing an optical material type.

    :param str name: Name of the material.
    :param float min_wl: Minimum applicable wavelength in ``default_unit``. Default: 0.
    :param float max_wl: Maximum applicable wavelength in ``default_unit``. Default: infinity.
    :param str default_unit: Unit of wavelength for dispersion formula and ``min_wl`` and ``max_wl``.
        Default: ``'um'``.
    """
    __slots__ = ('name', 'min_wl', 'max_wl', 'default_unit')
    _forbidden_name = ['', 'None', 'none', 'null']

    def __init__(self, name: str, min_wl: float = None, max_wl: float = None, default_unit: str = 'um'):
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
        #: Default unit.
        self.default_unit = default_unit

    def __repr__(self):
        return f'{self.__class__.__name__}({self._repr()})'

    @abc.abstractmethod
    def n(self, wavelength: Numeric) -> Numeric:
        """
        Computes refractive index.

        :param wavelength: Value of wavelength.
        :type: float or Tensor
        :return: Refractive index.
        :rtype: float or Tensor
        """
        pass

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        return {
            'type': self.__class__.__name__,
            'name': self.name,
            'min_wl': self.min_wl,
            'max_wl': self.max_wl,
            'default_unit': self.default_unit
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

    def _repr(self) -> str:
        return (f'name={self.name}, domain=('
                f'{base.Length.fmt(self.min_wl, self.default_unit)}, '
                f'{base.Length.fmt(self.max_wl, self.default_unit)})')

    def _make_wl(self, wl: Numeric) -> Numeric:
        wl = base.Length.default_to(wl, self.default_unit)
        m1, m2 = (wl.min().item(), wl.max().item()) if torch.is_tensor(wl) else (wl, wl)
        if m1 < self.min_wl * (1 - RANGE_CHECK_EPS) or m2 > self.max_wl * (1 + RANGE_CHECK_EPS):
            raise ValueError(
                f'Unsupported wavelength for material \'{self.name}\': '
                f'{wl}(unit: {self.default_unit})'
            )
        else:
            return wl


class Constant(Material):
    """
    Material with constant optical properties.

    :param float n: Constant refractive index.

    See :class:`Material` for descriptions for other parameters.
    """
    __slots__ = ('refractive_index',)

    def __init__(
        self,
        name: str,
        n: float,
        min_wl: float = None,
        max_wl: float = None,
        default_unit: str = 'um'
    ):
        super().__init__(name, min_wl, max_wl, default_unit)
        self.refractive_index: float = n  #: Refractive index.

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['n'] = self.refractive_index
        return d

    def _repr(self) -> str:
        return super()._repr() + f', n={utils.fmt(self.refractive_index)}'

    def n(self, wl: Numeric) -> Numeric:
        wl = self._make_wl(wl)
        n = self.refractive_index
        return torch.full_like(wl, n) if torch.is_tensor(wl) else n


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

    def __init__(
        self, name: str, a: float, b: float, c: float,
        min_wl: float = None, max_wl: float = None, default_unit: str = 'um'
    ):
        super().__init__(name, min_wl, max_wl, default_unit)
        self.a = a  #: :math:`A` in Cauchy formula.
        self.b = b  #: :math:`B` in Cauchy formula.
        self.c = c  #: :math:`C` in Cauchy formula.

    def n(self, wl: Numeric) -> Numeric:
        wl = self._make_wl(wl)
        iw2 = 1 / cast(Numeric, wl ** 2)
        n = (self.c * iw2 + self.b) * iw2 + self.a
        return n

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['a'] = self.a
        d['b'] = self.b
        d['c'] = self.c
        return d

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

    def __init__(
        self, name: str, coefficients: list[float],
        min_wl: float = None, max_wl: float = None, default_unit: str = 'um'
    ):
        super().__init__(name, min_wl, max_wl, default_unit)

        if len(coefficients) != 6:
            raise ValueError(f'Number of coefficients in Schott formula must be 6.')
        self.coefficients = coefficients  #: The six coefficients in Schott formula.

    def __getattr__(self, name: str):
        if len(name) == 2 and name[0] == 'a' and name[1].isdigit():
            return self.coefficients[int(name[1]) - 1]
        return super().__getattribute__(name)

    def n(self, wl: Numeric) -> Numeric:
        wl = self._make_wl(wl)
        wl2 = wl * wl
        iw2 = 1 / wl2
        iw4 = iw2 * iw2
        a = self.coefficients
        n2 = a[0] + a[1] * wl2 + a[2] * iw2 + a[3] * iw4 + a[4] * (iw4 * iw2) + a[5] * (iw4 * iw4)
        n = n2 ** 0.5
        return n

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['coefficients'] = self.coefficients
        return d

    def _repr(self) -> str:
        return f'{super()._repr()}, coefficients={_format_flist(self.coefficients)}'


class _Sellmeier(Material):
    __slots__ = ('ks', 'ls')
    _n_terms: int

    def __init__(
        self, name: str, ks: list[float], ls: list[float],
        min_wl: float = None, max_wl: float = None, default_unit: str = 'um'
    ):
        super().__init__(name, min_wl, max_wl, default_unit)

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

    def n(self, wl: Numeric) -> Numeric:
        wl = self._make_wl(wl)
        w2 = cast(Numeric, wl ** 2)
        n2 = 1 + sum([kc * w2 / (w2 - lc) for kc, lc in zip(self.ks, self.ls)])
        n = n2 ** 0.5
        return n

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['ks'] = self.ks
        d['ls'] = self.ls
        return d

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

    def __init__(
        self, name: str, a: float, b1: float, b2: float, wl1: float, wl2: float,
        min_wl: float = None, max_wl: float = None, default_unit: str = 'um'
    ):
        super().__init__(name, min_wl, max_wl, default_unit)
        self.a_pp = a + 1  #: :math:`A+1` in Sellmeier2 formula.
        self.b1 = b1  #: :math:`B_1` in Sellmeier2 formula.
        self.b2 = b2  #: :math:`B_2` in Sellmeier2 formula.
        self.swl1 = wl1 * wl1  #: :math:`\lambda_1^2` in Sellmeier2 formula.
        self.swl2 = wl2 * wl2  #: :math:`\lambda_2^2` in Sellmeier2 formula.

    def n(self, wl: Numeric) -> Numeric:
        wl = self._make_wl(wl)
        w2 = cast(Numeric, wl ** 2)
        n2 = self.a_pp + self.b1 * w2 / (w2 - self.swl1) + self.b2 / (w2 - self.swl2)
        n = n2 ** 0.5
        return n

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['a'] = self.a_pp - 1
        d['b1'] = self.b1
        d['b2'] = self.b2
        d['wl1'] = self.swl1 ** 0.5
        d['wl2'] = self.swl2 ** 0.5
        return d

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

    def __init__(
        self, name: str, a: float, b: float, c: float, d: float, e: float,
        min_wl: float = None, max_wl: float = None, default_unit: str = 'um'
    ):
        super().__init__(name, min_wl, max_wl, default_unit)
        self.a = a  #: :math:`A` in Sellmeier4 formula.
        self.b = b  #: :math:`B` in Sellmeier4 formula.
        self.c = c  #: :math:`C` in Sellmeier4 formula.
        self.d = d  #: :math:`D` in Sellmeier4 formula.
        self.e = e  #: :math:`E` in Sellmeier4 formula.

    def n(self, wl: Numeric) -> Numeric:
        wl = self._make_wl(wl)
        w2 = cast(Numeric, wl ** 2)
        n2 = self.a + self.b * w2 / (w2 - self.c) + self.d * w2 / (w2 - self.e)
        n = n2 ** 0.5
        return n

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        for k in ['a', 'b', 'c', 'd', 'e']:
            d[k] = getattr(self, k)
        return d

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

    def __init__(
        self, name: str, coefficients: list[float],
        min_wl: float = None, max_wl: float = None, default_unit: str = 'um'
    ):
        super().__init__(name, min_wl, max_wl, default_unit)

        if len(coefficients) != 6:
            raise ValueError(f'Number of coefficients in Herzberger formula must be 6.')
        self.coefficients = coefficients  #: The six coefficients in Herzberger formula.

    def __getattr__(self, name: str):
        if len(name) == 2 and name[0] == 'a' and name[1].isdigit():
            return self.coefficients[int(name[1]) - 1]
        return super().__getattribute__(name)

    def n(self, wl: Numeric) -> Numeric:
        wl = self._make_wl(wl)
        w2 = cast(Numeric, wl ** 2)
        m = 1 / (w2 - 0.028)
        _1, _2, _3, _4, _5, _6 = self.coefficients
        n = _1 + m * (_2 + m * _3) + w2 * (_4 + w2 * (_5 + w2 * _6))
        return n

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['coefficients'] = self.coefficients
        return d

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

    def __init__(
        self, name: str, n0: float, a: float, b: float,
        min_wl: float = None, max_wl: float = None, default_unit: str = 'um'
    ):
        super().__init__(name, min_wl, max_wl, default_unit)
        self.n0 = n0  #: :math:`n_0` in Conrady formula.
        self.a = a  #: :math:`a` in Conrady formula.
        self.b = b  #: :math:`b` in Conrady formula.

    def n(self, wl: Numeric) -> Numeric:
        wl = self._make_wl(wl)
        n = self.n0 + self.a / wl + self.b / cast(Numeric, wl ** 3.5)
        return n

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['n0'] = self.n0
        d['a'] = self.a
        d['b'] = self.b
        return d

    def _repr(self) -> str:
        return f'{super()._repr()}, n0={utils.fmt(self.n0)}, A={utils.fmt(self.a)}, B={utils.fmt(self.b)}'


class Air(Material):
    r"""
    Air in normal temperature and pressure, whose dispersion formula is:

    .. math::
        n=1+\left(6432.8+\frac{2949810}{146\lambda^2-1}+\frac{25540}{41\lambda^2-1}\right)10^{-8}

    See :class:`Material` for descriptions of parameters.
    """

    def n(self, wavelength: Numeric) -> Numeric:
        wl = self._make_wl(wavelength)
        n = _ref_n(wl * wl)
        return n


def _ref_n(wl2):
    return 1 + 6.4328e-5 + 2.94981e-2 * wl2 / (146 * wl2 - 1) + 2.5540e-4 * wl2 / (41 * wl2 - 1)


def get(name: str, default_none: bool = False) -> Union[Material, None]:
    """
    Get material by name from material library.

    :param str name: Name of the material.
    :param bool default_none: If true, return ``None`` when the material does not exist.
        Otherwise, an ``ValueError`` is raised.
    :return: Specified material.
    :rtype: Material
    """
    m = _lib.get(name, None)
    if m is None:
        if default_none:
            return None
        raise KeyError(f'Unknown material: {name}')
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
    return name in _lib


def search(pattern: str | re.Pattern) -> list[Material]:
    """
    Search materials by regular expression.

    :param str or re.Pattern pattern: The regular expression pattern.
    :return: List of materials whose names match the pattern.
    :rtype: list[Material]
    """
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    return [v for k, v in _lib.items() if pattern.search(k)]


def register(material: Material, exist_ok: bool = False):
    """
    Add a new class of material into material library.

    :param Material material: The material instance.
    :param bool exist_ok: If ``False``, raise :py:exc:`KeyError` if the material already exists.
        Otherwise, overwrite the existing material. Default: ``False``.
    """
    name = material.name
    if name in _lib and not exist_ok:
        raise KeyError(f'Material {name} already exists.')
    _lib[name] = material


def refractive_index(wavelength: Numeric, material: str) -> Numeric:
    """
    Compute refractive index for given wavelength and material.

    :param wavelength: Specified wavelength.
    :type: float or Tensor
    :param str material: Specified material.
    :return: Refractive index.
    :rtype: float or Tensor
    """
    m = get(material)
    n = m.n(wavelength)
    return n


def is_available(name: str) -> bool:
    """
    Check if given material is available in material library.

    :param str name: Name of the material.
    :return: If the material is available.
    :rtype: bool
    """
    return name in _lib


def list_all() -> list[str]:
    """
    List all available materials in material library.

    :return: List of the names of available materials.
    :rtype: list[str]
    """
    return list(_lib.keys())


def remove(name: str, ignore_if_absent: bool = False):
    """
    Remove a material from material library.

    :param str name: Name of the material.
    :param bool ignore_if_absent: If true, ignore material if it does not exist.
        A :py:exc:`KeyError` will be raised otherwise.
    :return: None
    """
    if name in _lib:
        del _lib[name]
    elif not ignore_if_absent:
        raise KeyError(f'Unknown material: {name}')


def update(name: str, material: Material):
    """
    Update a registered material.

    :param str name: Original name of the material.
    :param Material material: The new material instance.
    """
    if name in _lib:
        del _lib[name]
    if material.name in _lib:
        raise KeyError(f'Material {material.name} already exists.')
    register(material, exist_ok=True)


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
    materials = [m.to_dict() for m in _lib.values()]
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


_lib: dict[str, Material] = {}
with importlib.resources.open_text(__name__, 'builtin_materials.json') as f:
    load(f)
air: Air = cast(Air, _lib['air'])
vacuum: Constant = cast(Constant, _lib['vacuum'])
