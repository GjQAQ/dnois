import functools
import operator

import torch

from .. import AsJsonMixIn, typing as ty

__all__ = [
    'grid',
    'interval',

    'Grid',
    'Grid2d',
    'PixelGrid',
]


def _reshape(x: ty.Ts, n: int, idx: int) -> ty.Ts:
    x = x.reshape(*x.shape, *[1 for _ in range(n - 1)])
    x = x.transpose(idx - n, -n)
    return x


def interval(
    n: int, spacing: ty.Numeric = None, center: ty.Numeric = None, symmetric: bool = False, **kwargs
) -> ty.Ts:
    """
    Create a 1D evenly spaced grid.

    .. testsetup::

        from dnois import interval
        import torch
        torch.set_printoptions(precision=2)

    .. doctest::
        :options: +NORMALIZE_WHITESPACE

        >>> interval(3, 0.1)
        tensor([-0.10, 0.00, 0.10])
        >>> interval(3, torch.tensor([0.1, 0.2]))
        tensor([[-0.10, 0.00, 0.10],
                [-0.20, 0.00, 0.20]])
        >>> interval(3, torch.tensor([0.1, 0.2]), torch.tensor([-1, 1]))
        tensor([[-1.10, -1.00, -0.90],
                [ 0.80,  1.00,  1.20]])
        >>> interval(4)
        tensor([-2., -1., 0., 1.])
        >>> interval(4, symmetric=True)
        tensor([-1.50, -0.50, 0.50, 1.50])

    :param int n: Number of grid points.
    :param spacing: Spacing between grid points. If a tensor with shape ``(...)``,
        the returned tensor will have shape ``(..., n)``. Default: 1.
    :type spacing: Real or Tensor.
    :param center: Center of resulted grid points. If a tensor with shape ``(...)``,
        the returned tensor will have shape ``(..., n)``. Default: 0.
    :type center: Real or Tensor.
    :param bool symmetric: If ``True``, grid points are symmetric w.r.t. ``center``.
        Otherwise, ``n // 2`` points are smaller, ``n // 2 - 1`` points are larger
        and one point is ``center`` value. Only matters when ``n`` is even. Default: ``False``.
    :keyword kwargs: Tensor creation arguments passes to :py:func:`torch.linspace` like ``device``.
    :return: A tensor of shape ``(n,)`` if ``spacing`` and ``center`` are both scalars,
        otherwise of shape ``(*<broadcast shape of spacing and center>, n)``.
    :rtype: Tensor
    """
    if n % 2 == 1 or symmetric:
        x = torch.linspace(-(n - 1) / 2, (n - 1) / 2, n, **kwargs)
    else:
        x = torch.linspace(-n / 2, n / 2 - 1, n, **kwargs)
    if spacing is not None:
        if ty.is_scalar(spacing):
            x = x * spacing
        else:  # >1D tensor
            x = x * spacing.unsqueeze(-1)
    if center is not None:
        if ty.is_scalar(center):
            x = x + center
        else:
            x = x + center.unsqueeze(-1)
    return x


def grid(
    n: ty.Sequence[int],
    spacing: ty.Numeric | ty.Sequence[ty.Numeric] = None,
    center: ty.Numeric | ty.Sequence[ty.Numeric] = None,
    symmetric: bool = False,
    broadcast: bool = True,
    **kwargs
) -> list[ty.Ts]:
    """
    Create a ``len(n)``-D evenly spaced grid.

    .. testsetup::

        from dnois import grid
        import torch
        torch.set_printoptions(precision=2)

    .. doctest::
        :options: +NORMALIZE_WHITESPACE

        >>> grid((2, 3), 0.1)
        [tensor([[-0.10, -0.10, -0.10],
                 [ 0.00,  0.00,  0.00]]),
         tensor([[-0.10,  0.00,  0.10],
                 [-0.10,  0.00,  0.10]])]
        >>> grid((2, 3), 0.1, 1.)
        [tensor([[0.90, 0.90, 0.90],
                 [1.00, 1.00, 1.00]]),
         tensor([[0.90, 1.00, 1.10],
                 [0.90, 1.00, 1.10]])]
        >>> grid((2, 3), (0.1, 0.2), (-1., 1.))
        [tensor([[-1.10, -1.10, -1.10],
                 [-1.00, -1.00, -1.00]]),
         tensor([[0.80, 1.00, 1.20],
                 [0.80, 1.00, 1.20]])]
        >>> grid((2, 3), torch.tensor([0.1, 0.2]), torch.tensor([1., 2.]))
        [tensor([[[0.90, 0.90, 0.90],
                  [1.00, 1.00, 1.00]],
        <BLANKLINE>
                 [[1.80, 1.80, 1.80],
                  [2.00, 2.00, 2.00]]]),
         tensor([[[0.90, 1.00, 1.10],
                  [0.90, 1.00, 1.10]],
        <BLANKLINE>
                 [[1.80, 2.00, 2.20],
                  [1.80, 2.00, 2.20]]])]
        >>> grid((2, 3), symmetric=True)
        [tensor([[-0.50, -0.50, -0.50],
                 [ 0.50,  0.50,  0.50]]),
         tensor([[-1.,  0.,  1.],
                 [-1.,  0.,  1.]])]
        >>> grid((2, 3), broadcast=False)
        [tensor([[-1.],
                 [ 0.]]),
         tensor([[-1.,  0.,  1.]])]

    :param Sequence[int] n: Number of grid points in each dimension.
    :param spacing: Spacing between grid points in each dimension.
        A single ``Real`` or tensor indicates the spacing for all dimensions. Default: 1.
    :type spacing: Real or Tensor or Sequence[Real | Tensor]
    :param center: Center of resulted grid points in each dimension.
        A single ``Real`` or tensor indicates the center for all dimensions. Default: 0.
    :type center: Real or Tensor or Sequence[Real | Tensor]
    :param bool symmetric: See :py:func:`interval`. Default: ``False``.
    :param bool broadcast: Whether to broadcast resulted tensors. Default: ``True``.
    :param kwargs: Tensor creation arguments passes to :py:func:`torch.linspace`.
    :return: A list of ``len(n)``-D tensors if ``spacing`` and ``center`` are both scalars,
        otherwise of shape ``(*<broadcast shape of spacing and center>, *n)``.
        See above examples.
    :rtype: list[Tensor]
    """
    dims = len(n)
    if not isinstance(spacing, ty.Sequence):
        spacing = [spacing for _ in range(dims)]
    elif len(spacing) != dims:
        raise ValueError(f'Given dims={dims} but number of grid spacings is {len(spacing)}')
    if not isinstance(center, ty.Sequence):
        center = [center for _ in range(dims)]
    elif len(center) != dims:
        raise ValueError(f'Given dims={dims} but number of offsets is {len(center)}')

    g = [
        _reshape(interval(n[i], spacing[i], center[i], symmetric, **kwargs), dims, i)
        for i in range(dims)
    ]
    if broadcast:
        g = list(torch.broadcast_tensors(*g))
    return g


class Grid(AsJsonMixIn):
    """
    A class to represent a grid.

    See :func:`grid` for more details.
    """
    __slots__ = ('n', 'spacing', 'center', 'symmetric', '_lock')

    def __init__(
        self,
        n: ty.Sequence[int],
        spacing: ty.Numeric | ty.Sequence[ty.Numeric] = None,
        center: ty.Numeric | ty.Sequence[ty.Numeric] = None,
        symmetric: bool = False,
    ):
        if not isinstance(n, ty.Sequence) or not all(isinstance(n_item, int) for n_item in n):
            raise TypeError(f'A sequence of int expected for n')
        if len(n) == 0:
            raise ValueError('n can not be empty')
        if isinstance(spacing, ty.Sequence) and len(spacing) != len(n):
            raise ValueError(f'Given dims={len(n)} but number of grid spacings is {len(spacing)}')
        if isinstance(center, ty.Sequence) and len(center) != len(n):
            raise ValueError(f'Given dims={len(n)} but number of offsets is {len(center)}')

        self.n: tuple[int, ...] = tuple(n)  #: Number of grid points in each dimension.
        self.spacing: ty.Numeric | ty.Sequence[ty.Numeric] = spacing  #: Spacing between grid points in each dimension.
        self.center: ty.Numeric | ty.Sequence[ty.Numeric] = center  #: Center of resulted grid points in each dimension.
        self.symmetric: bool = symmetric  #: See :func:`interval`.
        self._lock = self

    def __setattr__(self, key, value):
        if hasattr(self, '_lock'):
            raise AttributeError(f'{type(self).__name__} object is immutable')
        else:
            return super().__setattr__(key, value)

    def __delattr__(self, item):
        if item == '_lock':
            raise RuntimeError(f'Cannot delete attribute {item}')
        else:
            return super().__delattr__(item)

    def size(self, dim: int = None) -> int:
        """
        Returns number of grid points in given dimension, or
        total number of grid points if ``dim`` is ``None``.

        :param int dim: Dimension index. Default: ``None``.
        :return: Number of grid points.
        :rtype: int
        """
        if dim is None:
            return functools.reduce(operator.mul, self.n)
        return self.n[dim]

    def span(self, dim: int = None) -> ty.Numeric:
        """
        Returns span of given dimension, or product of spans
        in all dimensions if ``dim`` is ``None``.
        By "span" we mean the distance between the first and last grid points
        in a dimension, which is ``(n[dim] - 1) * spacing[dim]``.
        Thus :attr:`.spacing` must be not ``None``.

        :param int dim: Dimension index. Default: ``None``.
        :return: Span of given dimension.
        :rtype: Real or Tensor
        :raises RuntimeError: If :attr:`spacing` is ``None``.
        """
        if self.spacing is None:
            raise RuntimeError('span not defined for a grid without spacing')
        if dim is None:
            return functools.reduce(operator.mul, map(self.span, range(self.ndim)))
        if isinstance(self.spacing, ty.Sequence):
            return (self.n[dim] - 1) * self.spacing[dim]
        else:
            return (self.n[dim] - 1) * ty.cast(ty.Ts, self.spacing)

    def vol(self, dim: int = None) -> ty.Numeric:
        """
        Returns volume of given dimension, or product of volumes
        in all dimensions if ``dim`` is ``None``.
        By "volume" we mean the width of the grids in one dimension
        when each points represent a volume element,
        which is ``n[dim] * spacing[dim]``.
        Thus :attr:`.spacing` must be not ``None``.

        :param int dim: Dimension index. Default: ``None``.
        :return: Volume of given dimension.
        :rtype: Real or Tensor
        :raises RuntimeError: If :attr:`spacing` is ``None``.
        """
        if self.spacing is None:
            raise RuntimeError('volume not defined for a grid without spacing')
        if dim is None:
            return functools.reduce(operator.mul, map(self.vol, range(self.ndim)))
        if isinstance(self.spacing, ty.Sequence):
            return self.n[dim] * self.spacing[dim]
        else:
            return self.n[dim] * ty.cast(ty.Ts, self.spacing)

    def make_points(self, broadcast: bool = True, **kwargs) -> list[ty.Ts]:
        """
        Create a list of tensors representing coordinates of grid points.
        See :func:`grid` for more details.
        """
        return grid(self.n, self.spacing, self.center, self.symmetric, broadcast, **kwargs)

    def to_dict(self, keep_tensor: bool = True) -> dict[str, ty.Any]:
        return {
            'n': self.n,
            'spacing': self._attr2dictitem('spacing', keep_tensor),
            'center': self._attr2dictitem('center', keep_tensor),
            'symmetric': self.symmetric,
        }

    @property
    def ndim(self):
        """
        Number of dimensions.

        :type: int
        """
        return len(self.n)


class Grid2d(Grid):
    """
    A subclass of :class:`Grid` whose ``ndim`` is always 2.
    """

    def __init__(
        self,
        n: ty.Sequence[int],
        spacing: ty.Numeric | ty.Sequence[ty.Numeric] = None,
        center: ty.Numeric | ty.Sequence[ty.Numeric] = None,
        symmetric: bool = False
    ):
        super().__init__(n, spacing, center, symmetric)
        if self.ndim != 2:
            raise ValueError(f'A 2D grid expected, got {self.ndim}D')


class PixelGrid(Grid):
    """
    A subclass of :class:`dnois.utils.Grid` but restricts
    the dimensionality to 2 and ``spacing`` to ``float`` s.
    This grid represents the pixel array of an imaging sensor.

    Both ``n`` and ``spacing`` can be either a 2-tuple
    (in horizontal and vertical directions, respectively)
    or a single value, in which case they are assumed to be the same.
    """

    def __init__(self, n: ty.Size2d, spacing: ty.Pair[float] = None):
        super().__init__(ty.size2d(n), ty.pair(spacing), symmetric=True)
        if self.ndim != 2:
            raise ValueError(f'A 2D grid expected, got {self.ndim}D')
        if not all(isinstance(s, float) for s in self.spacing):
            raise ValueError(f'All spacing values must be floats, got {self.spacing}')

    def to_dict(self, keep_tensor: bool = True) -> dict[str, ty.Any]:
        return {'n': self.n, 'spacing': self.spacing}

    @property
    def h(self) -> float:
        """Physical height of the sensor.\n\n:type: float"""
        return ty.cast(float, self.vol(0))

    @property
    def w(self) -> float:
        """Physical width of the sensor.\n\n:type: float"""
        return ty.cast(float, self.vol(1))

    @property
    def pixel_num(self) -> tuple[int, int]:
        """Alias for :attr:`.n`."""
        return ty.cast(tuple[int, int], self.n)

    @property
    def pixel_size(self) -> tuple[float, float]:
        """Alias for :attr:`.spacing`."""
        return ty.cast(tuple[float, float], self.spacing)
