"""
Package ``dnois`` provides some commonly used or basic functions and classes.
"""

from . import (
    camera,
    depth,
    ext,
    fourier,
    isp,
    mt,
    optics,
    scene,
    sensor,
    torch,
    utils
)
from .base import *
from .camera import *
from .torch.calc import *
from .utils import fmt

#: Default format to print float numbers.
float_print_fmt: str


def _():  # register customized attribute accessing behavior
    import sys
    from types import ModuleType

    from .base import unit

    class _Module(ModuleType):  # customized attribute accessing
        def __getattr__(self, name: str):
            if name == 'float_print_fmt':
                return unit.float_print_fmt
            else:
                return super().__getattr__(name)

        def __setattr__(self, key, value):
            if key == 'float_print_fmt':
                unit.float_print_fmt = value
            else:
                super().__setattr__(key, value)

    sys.modules[__name__].__class__ = _Module


_()
