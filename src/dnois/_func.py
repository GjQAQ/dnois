import math

from . import utils
from .base import typing as ty

__all__ = [
    'zernike',
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
    rs = [r ** m]
    r2 = rs[0] if m == 2 else r ** 2
    for i in range(m + 2, n + 1, 2):
        rs.append(rs[-1] * r2)
    return sum(c * r_power for c, r_power in zip(cs, rs))


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

    :param r: Radial distance.
    :param theta: Azimuthal angle.
    :param k: Index of Zernike polynomial.

    .. note::
        This definition of Zernike polynomials follows Noll's notation [#noll]_

    .. [#noll] Noll, R. J. (1976). Zernike polynomials and atmospheric turbulence.
        Journal of the Optical Society of America, 66(3), 207-211.
    """
    n, m = _nm_from_k(k)
    radial = _zernike_radial(r, n, m)
    if m == 0:
        return radial * (n + 1) ** 0.5

    if k % 2:  # odd
        azimuthal = utils.GenericCompute.sin(m * theta)
    else:
        azimuthal = utils.GenericCompute.cos(m * theta)
    return radial * azimuthal * (2 * n + 2) ** 0.5
