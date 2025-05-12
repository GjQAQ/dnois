import functools
import inspect

import torch

from ..base import typing, unit

__all__ = [
    'fmt',
    'invalid_option_msg',
    'with_external',

    'Conditional',
    'ExternalParamMixIn',
    'FixStateMixIn',
    'GenericCompute',
    'InfinityCond',
    'VarDict',
    'VarHook',
    'VarHookMixIn',
]


def fmt(v: float) -> str:
    """
    Format a ``float`` according to :data:`dnois.float_print_fmt`.

    :param float v: The ``float`` to be formatted.
    :return: Formatted string.
    :rtype: str
    """
    return unit.fmt(v)


_unset = object()


class ExternalParamMixIn:
    _external: list[str] = []

    def pick(self, name: str, value=None) -> typing.Any:
        """
        Determine the value of an :doc:`external parameter </content/guide/exparam>`
        ``name``. The return value is determined by the eponymous attribute of ``self``
        if ``value`` is ``None``, or ``value`` otherwise.

        :param str name: Name of the external parameter.
        :param Any value: Candidate of the external parameter.
            Default: the eponymous attribute of ``self``.
        :return: Value of the external parameter.
        :rtype: Any
        """
        picker = getattr(self, '_pick_' + name, None)
        if picker is not None:
            return picker(value)

        # This is needed even when value is not None to ensure name represents a valid config item
        attr = getattr(self, name, _unset)
        if attr is _unset:
            raise ValueError(f'Unknown external parameter for {self.__class__.__name__}: {name}')
        if value is None:
            return attr

        normalizer = getattr(self, '_normalize_' + name, None)
        return value if normalizer is None else normalizer(value)


def with_external(func: typing.Callable = None, *, exclude: str | typing.Sequence[str] = ()) -> typing.Callable:
    """
    A decorator to mark a method as having :doc:`external parameters </content/guide/exparam>`.
    The decorated method will be called with the actual value of the external parameters
    if no value is provided (specifically, ``None``).

    .. note::
        The default value of a parameter of a decorated function is omitted unless it is
        contained in ``exclude``.
    """

    def decorator(_func: typing.Callable, _exclude: typing.Sequence[str]) -> typing.Callable:
        @functools.wraps(_func)
        def wrapper(self, *args, **kwargs):
            externals = getattr(self, '_external', None)
            if externals is None:
                return _func(self, *args, **kwargs)

            sig = inspect.signature(_func)
            ba = sig.bind(self, *args, **kwargs)
            for pn in list(sig.parameters.keys()):
                if pn in externals and pn not in _exclude:
                    ba.arguments[pn] = self.pick(pn, ba.arguments.get(pn, None))
            return _func(*ba.args, **ba.kwargs)

        return wrapper

    if func is None:
        if isinstance(exclude, str):
            exclude = (exclude,)
        return functools.partial(decorator, _exclude=exclude)
    else:  # directly apply on a function without arguments
        return decorator(func, ())  # exclude is virtually the decorated function


class FixStateMixIn:  # warning: experimental
    _fixed_cache: dict[str, typing.Any] | None

    def fix(self):
        cache = getattr(self, '_fixed_cache', None)
        if cache is not None:
            return

        self._fixed_cache = {}

    def unfix(self):
        cache = getattr(self, '_fixed_cache', None)
        if cache is None:
            return

        self._fixed_cache = None

    @property
    def is_fixed(self) -> bool:
        return getattr(self, '_fixed_cache', None) is not None


_T = typing.TypeVar('_T')
VarHook = typing.Callable[[_T], _T | None]


class VarHookMixIn:  # to be documented
    _capture_hooks: dict[str, VarHook]

    def register_variable_hook(self, name: str, hook: VarHook):
        hooks = self._get_capture_hooks(True)
        if name in hooks:
            raise ValueError(f'Capture hook "{name}" for {self.__class__.__name__} already exists')
        hooks[name] = hook

    def remove_variable_hook(self, name: str):
        hooks = self._get_capture_hooks()
        if hooks is not None:
            hooks.pop(name, None)  # give default to avoid KeyError

    def variable_hook(self, name: str, obj: _T) -> _T:
        hooks = self._get_capture_hooks()
        if hooks is None:
            return obj

        hook = hooks.get(name, None)
        if hook is None:
            hook = hooks.get('*', None)
        if hook is not None:
            ret = hook(obj)
            if ret is not None:
                obj = ret
        return obj

    def _get_capture_hooks(self, create: bool = False) -> dict[str, VarHook]:
        hooks = getattr(self, '_capture_hooks', None)
        if hooks is None and create:
            self._capture_hooks = hooks = {}
        return hooks


class VarDict(dict[str, typing.Any]):
    def collector(self, name: str) -> VarHook:
        def hook(obj):
            self[name] = obj

        return hook


def invalid_option_msg(name: str, value, literals: type[typing.Literal] | list[str]) -> str:
    if not isinstance(literals, list):
        literals = typing.get_args(literals)
    return f'Unknown {name}: {value}, available: {", ".join(literals)}'


class Conditional:
    Cond = typing.Callable[[typing.Numeric], bool | torch.BoolTensor]
    Expr = typing.Callable[[typing.Numeric], typing.Any]

    def __init__(
        self,
        condition: Cond,
        expr_true: Expr,
        expr_false: Expr,
        condition_tensor: Cond = None,
        expr_true_tensor: Expr = None,
        expr_false_tensor: Expr = None,
    ):
        if condition_tensor is None:
            condition_tensor = condition
        if expr_true_tensor is None:
            expr_true_tensor = expr_true
        if expr_false_tensor is None:
            expr_false_tensor = expr_false

        self.condition = condition
        self.condition_tensor = condition_tensor
        self.expr_true = expr_true
        self.expr_false = expr_false
        self.expr_true_tensor = expr_true_tensor
        self.expr_false_tensor = expr_false_tensor

    def __call__(self, value: typing.Numeric) -> typing.Any:
        if torch.is_tensor(value):
            condition = self.condition_tensor(value)
            if condition.all():
                return self.expr_false_tensor(value)
            else:
                result = self.expr_true_tensor(value)
                if condition.any():
                    result = torch.where(condition, self.expr_false_tensor(value), result)
                return result
        else:
            if self.condition(value):
                return self.expr_false(value)
            else:
                return self.expr_true(value)


class InfinityCond(Conditional):
    def __init__(
        self,
        expr_finite: Conditional.Expr,
        expr_infinite: Conditional.Expr,
        expr_finite_tensor: Conditional.Expr = None,
        expr_infinite_tensor: Conditional.Expr = None
    ):
        super().__init__(
            lambda x: x == float('inf'), expr_finite, expr_infinite,
            lambda x: x.isinf(), expr_finite_tensor, expr_infinite_tensor
        )


class GenericCompute:
    def __init__(
        self,
        func: typing.Callable[[typing.Number], typing.Any] | typing.Callable[[typing.Ts | typing.Number], typing.Any],
        func_tensor: typing.Callable[[typing.Ts], typing.Any] = None,
    ):
        if func_tensor is None:
            func_tensor = func
        self.func = func
        self.func_tensor = func_tensor

    def __call__(self, value: typing.Number | typing.Ts) -> typing.Any:
        if torch.is_tensor(value):
            return self.func_tensor(value)
        else:
            return self.func(value)
