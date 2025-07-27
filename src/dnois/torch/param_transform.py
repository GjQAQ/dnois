from functools import partial

import torch
from torch import nn

from .. import utils
from ..base.serialize import AsJsonMixIn
from ..base.typing import Ts, Numeric, Any, Callable, overload, cast, tensor

__all__ = [
    'Composite',
    'Gt',
    'Lt',
    'ParamTransformModule',
    'Range',
    'Scale',
    'Transform',
]

TransFn = Callable[[Ts], Ts]
_unset = object()


def _ts_or_float(x: Numeric) -> Ts | float:
    if isinstance(x, float) or torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x)


class Transform(nn.Module, AsJsonMixIn):
    """
    Base class for parameter transformations (see :class:`~dnois.torch.ParamTransformModule`
    and :doc:`transformed parameters </content/guide/transform>`). This class also serves
    as a namespace containing functions to create commonly used transformations.
    All these function return a transformation object.

    The constructor of this class has two overloaded forms:

    - Accepts an inverse transformation after ``fn`` or as ``inverse`` argument;
    - Accepts any arguments combination (``*args`` and ``**kwargs``) which will be
      passed to ``fn`` along with latent value to calculate nominal value.
      No inverse transformation is specified in this case.

    In function descriptions below, :math:`x` indicates latent value and :math:`y`
    indicates nominal value.
    """

    @overload
    def __init__(self, fn: TransFn, inverse: TransFn = None):
        pass

    @overload
    def __init__(self, fn: TransFn, *args, **kwargs):
        pass

    def __init__(self, fn: TransFn, *args, **kwargs):
        super().__init__()
        if self.__class__ is Transform and fn is None:
            raise TypeError("Transform function fn must not be None")
        inverse = self._extract_inverse(*args, **kwargs)
        self._transform = partial(fn, *args, **kwargs) if inverse is None and fn is not None else fn
        self._inverse = inverse

    def forward(self, x: Ts) -> Ts:
        return self.transform(x)

    def transform(self, x: Ts) -> Ts:
        return self._transform(x)

    def inverse(self, y: Ts) -> Ts:
        return self._inverse(y)

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        raise NotImplementedError(f'General transformations cannot be converted to a dict')

    @property
    def invertible(self):
        return self._inverse is not None

    @classmethod
    def from_dict(cls, d: dict):
        if cls is not Transform:
            d.pop('type')
            return cls(**d)

        ty = d['type']
        subs = utils.subclasses(cls)
        for sub in subs:
            if sub.__name__ == ty:
                return cast(type[Transform], sub).from_dict(d)
        types = [sub.__name__ for sub in subs]
        raise RuntimeError(utils.invalid_option_msg('transform type', ty, types))

    @staticmethod
    def scale(s: Numeric) -> 'Transform':
        """
        Multiply the latent value by a scalar factor:

        .. math::
            y=sx,x=y/s

        :param s: Scale factor :math:`s`.
        :type s: float or Tensor
        """
        return Scale(s)

    @staticmethod
    def range(min_: Numeric, max_: Numeric) -> 'Transform':
        r"""
        Limit the range of parameter to :math:`(a,b)` using sigmoid function:

        .. math::
            y=(b-a)\sigma(x)+a,x=\sigma^{-1}(\frac{y-a}{b-a})

        where :math:`\sigma(t)=\frac{1}{1+\e^{-t}}` and :math:`\sigma^{-1}(t)=\ln\frac{t}{1-t}`.

        :param min_: Lower bound :math:`a`.
        :type min_: float or Tensor
        :param max_: Upper bound :math:`b`.
        :type max_: float or Tensor
        """
        return Range(min_, max_)

    @staticmethod
    def positive() -> 'Transform':
        r"""
        Force parameter to be positive using exponential function:

        .. math::
            y=\e^x,x=\ln y
        """
        return Gt()

    @staticmethod
    def negative() -> 'Transform':
        r"""
        Force parameter to be negative using exponential function:

        .. math::
            y=-\e^x,x=\ln -y
        """
        return Lt()

    @staticmethod
    def gt(limit: Numeric) -> 'Transform':
        r"""
        Force parameter to be greater than :math:`a` using exponential function:

        .. math::
            y=a+\e^x,x=\ln(y-a)

        :param limit: Lower bound :math:`a`.
        :type limit: float or Tensor
        """
        return Gt(limit)

    @staticmethod
    def lt(limit: Numeric) -> 'Transform':
        r"""
        Force parameter to be less than :math:`b` using exponential function:

        .. math::
            y=b-\e^x,x=\ln(b-y)

        :param limit: Upper bound :math:`b`.
        :type limit: float or Tensor
        """
        return Lt(limit)

    @staticmethod
    def composite(*transforms: 'Transform') -> 'Transform':
        r"""
        Chain some transformations into a single transformation.
        Resulted transformation is to apply all the transformations sequentially
        and resulted inverse transformation is to apply all the inversions in a reversed order.

        :param Transform transforms: One or more transformations to apply.
        """
        return Composite(*transforms)

    @staticmethod
    def _extract_inverse(*args, **kwargs):
        if 'inverse' in kwargs:
            if len(args) + len(kwargs) > 1:
                raise ValueError(f'No more arguments is acceptable when inverse is given')
            return kwargs['inverse']
        elif len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return args[0]
        else:
            return None


class Scale(Transform):
    invertible = True

    def __init__(self, s: Numeric):
        super().__init__(cast(Callable, None))
        self.register_buffer('s', None)
        self.s = tensor(s)

    def transform(self, x: Ts) -> Ts:
        return x * self.s

    def inverse(self, y: Ts) -> Ts:
        return y / self.s

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        return {
            'type': self.__class__.__name__,
            's': self._attr2dictitem('s', keep_tensor)
        }


class Range(Transform):
    invertible = True

    def __init__(self, min_: Numeric, max_: Numeric):
        super().__init__(cast(Callable, None))
        self.register_buffer('min', None)
        self.register_buffer('range_', None)  # conflict with range() method in super class
        self.min = tensor(min_)
        self.range_ = tensor(max_) - self.min

    def transform(self, x: Ts) -> Ts:
        return self.min + self.range_ * x.sigmoid()

    def inverse(self, y: Ts) -> Ts:
        return torch.logit((y - self.min) / self.range_)

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        _min = self._attr2dictitem('min', keep_tensor)
        return {
            'type': self.__class__.__name__,
            'min_': _min,
            'max_': self._attr2dictitem('range', keep_tensor) + _min,
        }


class Gt(Transform):
    invertible = True

    def __init__(self, limit: Numeric = None):
        super().__init__(cast(Callable, None))
        self.register_buffer('limit', None)
        self.limit = tensor(limit)

    def transform(self, x: Ts) -> Ts:
        return x.exp() if self.limit is None else self.limit + x.exp()

    def inverse(self, y: Ts) -> Ts:
        return y.log() if self.limit is None else torch.log(y - self.limit)

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        return {
            'type': self.__class__.__name__,
            'limit': self._attr2dictitem('limit', keep_tensor),
        }


class Lt(Transform):
    invertible = True

    def __init__(self, limit: Numeric = None):
        super().__init__(cast(Callable, None))
        self.register_buffer('limit', None)
        self.limit = tensor(limit)

    def transform(self, x: Ts) -> Ts:
        return -x.exp() if self.limit is None else self.limit - x.exp()

    def inverse(self, y: Ts) -> Ts:
        return y.neg().log() if self.limit is None else torch.log(self.limit - y)

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        return {
            'type': self.__class__.__name__,
            'limit': self._attr2dictitem('limit', keep_tensor),
        }


class Composite(Transform):
    def __init__(self, *transforms: Transform):
        super().__init__(cast(Callable, None))
        self.transforms = nn.ModuleList(transforms)

    def transform(self, x: Ts) -> Ts:
        for t in self.transforms:
            x = t.transform(x)
        return x

    def inverse(self, y: Ts) -> Ts:
        for t in reversed(self.transforms):
            y = t.inverse(y)
        return y

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        return {
            'type': self.__class__.__name__,
            'transforms': [t.to_dict(keep_tensor) for t in self.transforms]
        }

    @property
    def invertible(self):
        return all(m.invertible for m in self.transforms)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(*[Transform.from_dict(item) for item in d['transforms']])


class ParamTransformModule(nn.Module):
    """
    A subclass of :class:`torch.nn.Module` that can create and manage
    :doc:`transformed parameters </content/guide/transform>`.

    Transformed parameter can be registered given either its nominal value
    (by calling overridden :meth:`.register_parameter`) or latent value (by calling
    :meth:`.register_latent_parameter`). Note that in first case an inverse transformation
    must be given to compute the latent value. If a parameter with same name has been
    registered already, it will be automatically converted to a transformed parameter.
    Specifying a transformation for existent parameter by calling :meth:`.set_transform`
    has same effect. Transformations are specified in the form of a :class:`Transform` instance.

    .. note::
        **Implementation Detail** The latent value of each transformed parameter is
        a :class:`torch.nn.Parameter` attribute with name like ``_latent_<param>``.
    """
    _param_transforms: nn.ModuleDict  # dict[str, Transform]

    def __getattr__(self, name: str):
        param_transforms = self._modules.get('_param_transforms', {})
        if name in param_transforms:  # if a transformed param, return its nominal value
            latent_name = self._latent_name(name)
            return param_transforms[name](super().__getattr__(latent_name))
        else:
            return super().__getattr__(name)

    def __setattr__(self, name, value):
        param_transforms = self._modules.get('_param_transforms', {})
        if param_transforms is _unset:  # fall back if no param transform set
            return super().__setattr__(name, value)

        if name in param_transforms:  # if a transformed param, set its latent value
            latent_name = self._latent_name(name)
            if value is None:
                return super().__setattr__(latent_name, value)
            transform = param_transforms[name]
            if not transform.invertible:
                raise RuntimeError(f'No inverse transformation specified for transformed parameter {name} '
                                   f'so it cannot be assigned. Assign to its latent "{latent_name}" '
                                   f'if you intend to modify its value directly.')
            else:
                latent_param = transform.inverse(value)
                if not isinstance(latent_param, nn.Parameter):
                    latent_param = nn.Parameter(latent_param)
                super().__setattr__(latent_name, latent_param)
        else:
            return super().__setattr__(name, value)

    def __delattr__(self, name):
        param_transforms = self._modules.get('_param_transforms', {})
        if name in param_transforms:  # if a transformed param, delete its latent value and transformation
            latent_name = self._latent_name(name)
            super().__delattr__(latent_name)
            del param_transforms[name]
        else:
            return super().__delattr__(name)

    def register_parameter(
        self, name: str, param: nn.Parameter | None, transform: Transform = None
    ) -> None:
        """
        Similar to :meth:`torch.nn.Module.register_parameter`, but allows you to register a transformed
        parameter as long as ``transform`` is given.

        If ``name`` corresponds to a vanilla parameter (i.e. not transformed parameter)
        but ``transform`` is given, it will be converted to a transformed one.

        :param str name: Name of the parameter.
        :param Parameter param: Nominal :class:`torch.nn.Parameter` instance to be registered.
        :param Transform transform: Transformation object.
        """
        if transform is None:
            return super().register_parameter(name, param)
        if not transform.invertible:
            raise ValueError(f'Inverse transformation cannot be None when registering parameter {name} '
                             f'with transformation given. Call register_latent_parameter to register '
                             f'a transformed parameter without inverse transformation.')

        if param is not None:
            param.data = transform.inverse(param.data)
        return self.register_latent_parameter(name, param, transform)

    def register_latent_parameter(
        self, name: str, param: nn.Parameter | None, transform: Transform
    ):
        """
        Similar to :meth:`.register_parameter`, but takes as input the latent value
        rather than nominal value. In this way, the inverse transformation need not be
        provided since the initial latent value is known.

        If ``name`` corresponds to a vanilla parameter (i.e. not transformed parameter)
        it will be converted to a transformed one.

        :param str name: Name of the parameter.
        :param Parameter param: Latent :class:`torch.nn.Parameter` instance to be registered.
        :param Transform transform: Transformation object.
        """
        lt_name = self._latent_name(name)
        param_obj = getattr(self, name, _unset)
        latent_param_obj = getattr(self, lt_name, _unset)
        if param_obj is not _unset:
            if isinstance(latent_param_obj, nn.Parameter):
                delattr(self, lt_name)
            elif not isinstance(param_obj, nn.Parameter):
                raise AttributeError(
                    f'Attribute {name} of class {self.__class__.__name__} is not a parameter'
                )
            else:
                delattr(self, name)

        super().register_parameter(lt_name, param)

        transforms = self._get_transforms_dict()
        transforms[name] = transform

    def set_transform(self, name: str, transform: Transform):
        """
        Set transformation for parameter ``name``.

        If ``name`` corresponds to a vanilla parameter (i.e. not transformed parameter)
        it will be converted to a transformed one.

        :param str name: Name of the parameter.
        :param Transform transform: Transformation object.
        """
        return self.register_parameter(name, nn.Parameter(getattr(self, name)), transform)

    def remove_transform(self, name: str):
        """
        Remove transformation for parameter ``name``.

        :param str name: Name of the parameter.
        """
        param_transforms = self._modules.get('_param_transforms', _unset)
        if param_transforms is _unset:
            raise RuntimeError(f'No parameter transform set in {self.__class__.__name__}')
        if name not in param_transforms:
            raise RuntimeError(f'No parameter transform set for {name} in {self.__class__.__name__}')

        nominal_value = getattr(self, name)
        delattr(self, name)
        if not isinstance(nominal_value, nn.Parameter):
            nominal_value = nn.Parameter(nominal_value)
        self.register_parameter(name, nominal_value)

    @property
    def nominal_values(self) -> dict[str, Ts]:
        """
        A ``dict`` whose keys are names of all parameters of this module
        and values are their values. The values are nominal ones for transformed parameters.

        :type: dict[str, Tensor]
        """
        named_params = {k: v for k, v in self.named_parameters(recurse=False)}
        param_transforms = getattr(self, '_param_transforms', _unset)
        if param_transforms is _unset:
            return named_params

        for k in list(named_params.keys()):
            if k.startswith('_latent_'):
                name = self._nominal_name(k)
                if name in param_transforms:
                    value = named_params.pop(k)
                    named_params[name] = param_transforms[name](value)
        return named_params

    @property
    def transformed_parameters(self) -> dict[str, tuple[torch.nn.Parameter, Transform]]:
        """
        A ``dict`` whose keys are names of all transformed parameters of this module.
        The value corresponding to each key is a tuple containing:

        - The latent value, a :class:`torch.nn.Parameter` instance;
        - Corresponding transformation object.

        :type: dict[str, tuple[Parameter, Transform]]
        """
        param_transforms = getattr(self, '_param_transforms', {})
        return {
            name: (super().__getattr__(self._latent_name(name)), tr)
            for name, tr in param_transforms.items()
        }

    def _get_transforms_dict(self):
        if '_param_transforms' not in self._modules:
            self._param_transforms = nn.ModuleDict()
        return self._modules['_param_transforms']

    @staticmethod
    def _latent_name(nominal_name: str) -> str:
        return f'_latent_{nominal_name}'

    @staticmethod
    def _nominal_name(latent_name: str) -> str:
        return latent_name[8:]
