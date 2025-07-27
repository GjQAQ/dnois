from functools import partial, wraps
import numbers

import torch
from torch import nn

from .param_transform import *
from ..base import typing
from ..base.serialize import AsJsonMixIn
from ..base.typing import Callable, Ts, cast

__all__ = [
    'DeviceMixIn',
    'DtypeMixIn',
    'EnhancedModule',
    'FreezeParamMixIn',
    'TensorAsDelegate',
    'TensorContainerMixIn',
    'WrapperModule',
]


class WrapperModule(nn.Module):
    """
    A class to wrap a function as a :py:class:`torch.nn.Module`.

    .. doctest::
        :skipif: True

        >>> s = WrapperModule(torch.sum, dim=(-2, -1))
        >>> x = torch.rand(4)
        >>> s(x)  # equivalent to torch.sum(x, dim=(-2, -1))

    :param Callable func: The function to be wrapped.
    :param args: Positional arguments to be passed to ``func`` when this module is called.
    :param kwargs: Keyword arguments to be passed to ``func`` when this module is called.
    """

    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__()
        self._impl = partial(func, *args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Call the wrapped function ``func``.

        :param args: Additional positional arguments to be passed to ``func``.
        :param kwargs: Additional keyword arguments to be passed to ``func``.
        :return: The returned value of the wrapped function.
        :rtype: Any
        """
        return self._impl(*args, **kwargs)


_unset = object()


def _check_consistency(attr: str, obj, ts: Ts, error: bool) -> bool:
    v1, v2 = getattr(obj, attr), getattr(ts, attr)
    if v1 != v2:
        if error:
            raise RuntimeError(f'{attr.capitalize()} mismatch: {v1} for an instance of '
                               f'{obj.__class__.__name__} while {v2} for an incoming tensor')
        else:
            return False
    return True


class TensorAsDelegate:
    def new_tensor(self, data, **kwargs) -> Ts:
        return self._delegate().new_tensor(data, **kwargs)

    def new_full(self, size, fill_value, **kwargs) -> Ts:
        return self._delegate().new_full(size, fill_value, **kwargs)

    def new_empty(self, size, **kwargs) -> Ts:
        return self._delegate().new_empty(size, **kwargs)

    def new_ones(self, size, **kwargs) -> Ts:
        return self._delegate().new_ones(size, **kwargs)

    def new_zeros(self, size, **kwargs) -> Ts:
        return self._delegate().new_zeros(size, **kwargs)

    def arange(self, *args, **kwargs) -> Ts:
        d = self._delegate()
        return torch.arange(*args, **kwargs, device=d.device, dtype=d.dtype)

    def linspace(self, *args, **kwargs) -> Ts:
        d = self._delegate()
        return torch.linspace(*args, **kwargs, device=d.device, dtype=d.dtype)

    def rand(self, *args, **kwargs) -> Ts:
        d = self._delegate()
        return torch.rand(*args, **kwargs, device=d.device, dtype=d.dtype)

    def randn(self, *args, **kwargs) -> Ts:
        d = self._delegate()
        return torch.randn(*args, **kwargs, device=d.device, dtype=d.dtype)

    def _delegate(self) -> Ts:
        if not isinstance(self, nn.Module):
            raise NotImplementedError(f'A subclass of {TensorAsDelegate.__name__} that is not derived from '
                                      f'torch.nn.Module must implement {self._delegate.__name__} method')
        # register stub dynamically to avoid calling __init__
        if not hasattr(self, '_delegate_tensor'):
            t = None
            for p in self.parameters():
                t = p
                break
            if t is None:
                for b in self.buffers():
                    t = b
                    break
            if t is None:
                t = torch.tensor([])
            self.register_buffer('_delegate_tensor', t.new_tensor([]), False)
        return self._delegate_tensor


class DeviceMixIn(TensorAsDelegate):
    """
    Some :py:class:`torch.Tensor` s may be associated to objects of the class
    (e.g. buffers and parameters of :py:class:`torch.nn.Module`)
    derived from this class. They are assumed to be on the same device,
    which is the value of :attr:`device`.
    """

    def _check_consistency(self, ts: Ts, error: bool = True) -> bool:
        return _check_consistency('device', self, ts, error)

    def _cast(self, ts: Ts) -> Ts:
        return ts.to(device=self.device)

    @property
    def device(self) -> torch.device:
        """
        Device of this object.

        :type: :py:class:`torch.device`
        """
        dlg = self._delegate()
        # torch.get_default_device() is not available for old versions
        return torch.tensor(0.).device if dlg is None else dlg.device


class DtypeMixIn(TensorAsDelegate):
    """
    Some :py:class:`torch.Tensor` s may be associated to objects of the class
    (e.g. buffers and parameters of :py:class:`torch.nn.Module`)
    derived from this class. They are assumed to have same data type,
    which is the value of :attr:`dtype`.
    """

    def _check_consistency(self, ts: Ts, error: bool = True) -> bool:
        return _check_consistency('dtype', self, ts, error)

    def _cast(self, ts: Ts) -> Ts:
        return ts.to(dtype=self.dtype)

    @property
    def dtype(self) -> torch.dtype:
        """
        Data type of this object.

        :type: :py:class:`torch.dtype`
        """
        dlg = self._delegate()
        return torch.get_default_dtype() if dlg is None else dlg.dtype


class TensorContainerMixIn(DeviceMixIn, DtypeMixIn):
    def _check_consistency(self, ts: Ts, error: bool = True) -> bool:
        return (_check_consistency('device', self, ts, error) and
                _check_consistency('dtype', self, ts, error))

    def _cast(self, ts: Ts) -> Ts:
        return ts.to(device=self.device, dtype=self.dtype)


class FreezeParamMixIn(nn.Module):
    def freeze(self, name: str | typing.Sequence[str] = None):
        """Equivalent to ``self.set_optimizable(name, False)``. See :meth:`.set_optimizable`."""
        self.set_optimizable(name, False)

    def unfreeze(self, name: str | typing.Sequence[str] = None):
        """Equivalent to ``self.set_optimizable(name, True)``. See :meth:`.set_optimizable`."""
        self.set_optimizable(name, True)

    def set_optimizable(self, name: str | typing.Sequence[str] = None, optimizable: bool = True):
        """
        Specify whether a parameter is optimizable.

        :param str name: Name of the parameter. If ``None``, all parameters will be set.
            It follows the same convention as :meth:`torch.nn.Module.get_parameter`.
        :param bool optimizable: Whether the specified parameter is optimizable. Default: ``True``.
        """
        if name is None:
            for p in self.parameters():
                p.requires_grad = optimizable
        elif isinstance(name, str):
            param = self.get_parameter(name)
            param.requires_grad = optimizable
        else:
            for n in name:
                param = self.get_parameter(n)
                param.requires_grad = optimizable


class EnhancedModule(
    ParamTransformModule,
    AsJsonMixIn,
    TensorContainerMixIn,
    FreezeParamMixIn,
):
    _writable_params = set()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_to_dict = cls.__dict__.get('to_dict', None)
        original_from_dict = cls.__dict__.get('from_dict', None)

        if original_to_dict is not None:
            original_to_dict = cast(Callable, original_to_dict)

            def _wrapped_to_dict(self: EnhancedModule, keep_tensor: bool = True):
                d = original_to_dict(self, keep_tensor)
                tp = self.transformed_parameters
                if len(tp) != 0:
                    d['transformed_parameters'] = {k: (
                        self._attr2dictitem(self._latent_name(k), keep_tensor),
                        v[1].to_dict(keep_tensor)
                    ) for k, v in tp.items()}
                return d

            cls.to_dict = wraps(original_to_dict)(_wrapped_to_dict)

        if original_from_dict is not None:
            if isinstance(original_from_dict, classmethod):
                original_from_dict = original_from_dict.__wrapped__

            def _wrapped_from_dict(clz: type[EnhancedModule], d: dict):
                tp = d.pop('transformed_parameters', {})
                obj = original_from_dict(clz, d)
                for k, (param, transform) in tp.items():
                    if not isinstance(param, nn.Parameter):
                        param = nn.Parameter(torch.tensor(param))
                    obj.register_latent_parameter(k, param, Transform.from_dict(transform))
                return obj

            cls.from_dict = classmethod(wraps(original_from_dict)(_wrapped_from_dict))

    def __setattr__(self, key, value):
        # allows modification to parameters whose value is None or scalar tensor by attribute assignment
        # rather than fussy register_parameter call
        params: dict | object = self.__dict__.get('_parameters', _unset)
        if params is _unset:
            return super().__setattr__(key, value)

        if key in self._writable_params or (key in params and (params[key] is None or params[key].ndim == 0)):
            self.set_scalar_param(key, value)
        else:
            super().__setattr__(key, value)

    def set_scalar_param(self, name: str, value: numbers.Number | Ts):
        if not torch.is_tensor(value):
            if not isinstance(value, numbers.Number):
                raise TypeError(f'Value of parameter {name} of {type(self).__name__} must be a number')
            value = self.new_tensor(value)
        if not isinstance(value, nn.Parameter):
            value = nn.Parameter(value.to(device=self.device, dtype=self.dtype))
        # it is handled correctly when key refers to a transformed parameter
        super().__setattr__(name, value)
