import torch

from .check import requires

__all__ = [
    'mpl_available',
    'visfunc',
]


def mpl_available() -> bool:
    """
    Return whether matplotlib is available.

    :return: ``True`` if matplotlib is available, ``False`` otherwise.
    :rtype: bool
    """
    try:
        import matplotlib
        return True
    except ImportError:
        return False


def visfunc(func):
    return requires('matplotlib')(torch.no_grad()(func))
