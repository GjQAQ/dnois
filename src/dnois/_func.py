import math

from . import utils, torch as _t
from .base import typing as ty

__all__ = [
    'zernike',
    'zernike_cpd',
]


def _nm_from_k(k: int) -> tuple[int, int]:
    if k <= 0:
        raise ValueError(f'k must be positive, got {k}')

    n = 0
    while (n + 1) * (n + 2) // 2 < k:
        n += 1

    idx = k - n * (n + 1) // 2
    if n % 2 == 1:  # odd
        m = (idx + 1) // 2 * 2 - 1
    else:
        m = idx // 2 * 2
    return n, m


def _zernike_radial_coefficient(n: int, m: int, s: int):
    c = math.comb(n - s, s) * math.comb(n - 2 * s, (n + m) // 2 - s)
    return -c if s % 2 else c


def _zernike_radial(r: ty.Numeric, n: int, m: int) -> ty.Numeric:
    # lower order to higher order
    cs = [_zernike_radial_coefficient(n, m, s) for s in range((n - m) // 2, -1, -1)]

    r2 = r ** 2
    if m == 0:
        if n == 0:
            # when n==0, 1. is returned no matter r is float or tensor without this branch
            return utils.GenericCompute.one(r)
        return _t.polynomial(r2, cs)
    elif m == 1:
        rm = r
    elif m == 2:
        rm = r2
    else:
        rm = r ** m

    value = rm * _t.polynomial(r2, cs)
    return value


def _zernike_radial_over_r(r: ty.Numeric, n: int, m: int) -> ty.Numeric:
    # this function is used to compute Cartesian derivatives of zernike
    if m == 0:
        raise RuntimeError(f'Unexpected call to {_zernike_radial_over_r.__name__}')

    cs = [_zernike_radial_coefficient(n, m, s) for s in range((n - m) // 2, -1, -1)]
    r2 = r ** 2
    if m == 1:
        if n == 1:
            # see comment in _zernike_radial
            return utils.GenericCompute.one(r) * cs[0]
        return _t.polynomial(r2, cs)
    elif m == 2:
        factor = r
    elif m == 3:
        factor = r2
    else:
        factor = r ** (m - 1)
    value = factor * _t.polynomial(r2, cs)
    return value


def _zernike_radial_derivative(r: ty.Numeric, n: int, m: int) -> ty.Numeric:
    # lower order to higher order
    cs = [_zernike_radial_coefficient(n, m, s) * (n - 2 * s) for s in range((n - m) // 2, -1, -1)]

    r2 = r ** 2
    if m == 0:
        if n == 0:
            # see comment in _zernike_radial
            return utils.GenericCompute.zero(r)
        return r * _t.polynomial(r2, cs[1:])
    elif m == 1:
        if n == 1:
            return utils.GenericCompute.one(r) * cs[0]
        return _t.polynomial(r2, cs)

    if m == 2:
        factor = r
    elif m == 3:
        factor = r2
    else:
        factor = r ** (m - 1)
    value = factor * _t.polynomial(r2, cs)
    return value


def _zernike_normalization(n: int, m: int) -> ty.Numeric:
    if m == 0:
        return (n + 1) ** 0.5
    return (2 * (n + 1)) ** 0.5


def zernike(r: ty.Numeric, theta: ty.Numeric, k: int) -> ty.Numeric:
    r"""
    Computes the :math:`k`-th term of Zernike polynomials:

    .. math::
        Z_k(r,\theta)=\left\{\begin{array}{ll}
            \sqrt{n+1}R_n^0(r),&m=0,\\
            \sqrt{2(n+1)}R_n^m(r)\sin(m\theta),&m \text{is odd},\\
            \sqrt{2(n+1)}R_n^m(r)\cos(m\theta),&m \text{else},
        \end{array}\right.\\
        R_n^m(r)=\sum_{s=0}^{(n-m)/2}\frac{(-1)^s(n-s)!}
        {s!\left((n+m)/2-s\right)!\left((n-m)/2-s\right)!}r^{n-2s}

    where :math:`n` and :math:`m` are determined by :math:`k` as follows:

    === === ===
    k   n   m
    === === ===
    1   0   0
    2   1   1
    3   1   1
    4   2   0
    5   2   2
    6   2   2
    7   3   1
    ...
    -----------
    === === ===

    .. note::
        This definition of Zernike polynomials follows Noll's notation [#noll]_

    :param r: Radial distance.
    :param theta: Azimuthal angle.
    :param k: Index of Zernike polynomial.
    :return: Value of :math:`Z_k(r,\theta)`.

    .. [#noll] Noll, R. J. (1976). Zernike polynomials and atmospheric turbulence.
        Journal of the Optical Society of America, 66(3), 207-211.
    """
    n, m = _nm_from_k(k)
    radial = _zernike_radial(r, n, m)
    norm = _zernike_normalization(n, m)
    if m == 0:
        return radial * norm

    if k % 2:  # odd
        azimuthal = utils.GenericCompute.sin(m * theta)
    else:
        azimuthal = utils.GenericCompute.cos(m * theta)
    return radial * azimuthal * norm


def zernike_cpd(r: ty.Numeric, theta: ty.Numeric, k: int) -> tuple[ty.Numeric, ty.Numeric]:
    r"""
    Computes the Cartesian partial derivatives of :math:`k`-th term of Zernike polynomials.
    See :func:`zernike` for more details.

    :param r: Radial distance.
    :param theta: Azimuthal angle.
    :param k: Index of Zernike polynomial.
    :return: A 2-tuple indicating :math:`\pfrac{Z_k(r,\theta)}{x}` and
        :math:`\pfrac{Z_k(r,\theta)}{y}`.
    """
    n, m = _nm_from_k(k)
    d_radial = _zernike_radial_derivative(r, n, m)
    norm = _zernike_normalization(n, m)
    if m == 0:
        _1 = d_radial * norm
        return _1 * utils.GenericCompute.cos(theta), _1 * utils.GenericCompute.sin(theta)

    if k % 2:  # odd
        azimuthal = utils.GenericCompute.sin(m * theta)
        d_azimuthal = m * utils.GenericCompute.cos(m * theta)
    else:
        azimuthal = utils.GenericCompute.cos(m * theta)
        d_azimuthal = -m * utils.GenericCompute.sin(m * theta)
    _1 = d_radial * azimuthal
    _2 = _zernike_radial_over_r(r, n, m) * d_azimuthal
    dx = _1 * utils.GenericCompute.cos(theta) - _2 * utils.GenericCompute.sin(theta)
    dy = _1 * utils.GenericCompute.sin(theta) + _2 * utils.GenericCompute.cos(theta)
    return dx * norm, dy * norm
