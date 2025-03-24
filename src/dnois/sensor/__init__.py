"""
:mod:`dnois.sensor` implements image sensor models with frequently used functions,
such as Bayer CFA, noise simulation, Gamma correction, quantization, saturation, etc.
"""
from ._func import *
from ._main import *
from .noise import *

from . import noise
