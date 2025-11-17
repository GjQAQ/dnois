import collections.abc
import contextlib
import functools
import inspect
import math

import torch

from ..base import typing, unit

__all__ = [
    'context_cache',
    'enable_group_cache',
    'fmt',
    'invalid_option_msg',
    'type_normalizer',
    'with_external',

    'ContextCache',
    'Conditional',
    'ConvertSetAttrMixIn',
    'Exparam',
    'ExternalParamMixIn',
    'FixStateMixIn',
    'GenericCompute',
    'HookRemover',
    'InfinityCond',
    'VarDict',
    'VarHook',
    'VarHookMixIn',
]

_unset = object()
Exparam: type = type('Exparam', (), {})


def fmt(v: float) -> str:
    """
    Format a ``float`` according to :attr:`~dnois.base.conf.float_print_fmt`.

    .. seealso::
        :meth:`dnois.Unit.fmt` formats values with unit.

    :param float v: The ``float`` to be formatted.
    :return: Formatted string.
    :rtype: str
    """
    return unit.fmt(v)


class ExternalParamMixIn:
    _external: list[str] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._external = cls._external.copy()  # assign to avoid shared list

        annotations = inspect.get_annotations(cls)
        for k, v in annotations.items():
            if v is Exparam:
                cls._external.append(k)

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


class ConvertSetAttrMixIn(ExternalParamMixIn):
    def __setattr__(self, key: str, value):
        normalizer = getattr(self, '_normalize_' + key, None)
        if normalizer is not None:
            value = normalizer(value)
        return super().__setattr__(key, value)


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


def type_normalizer(t: type) -> typing.Callable:
    def normalizer(v):
        if isinstance(v, t):
            return v
        return t.create(v)  # noqa

    return normalizer


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


class HookRemover:
    def __init__(self, hook: VarHook, hook_list: list[VarHook]):
        self.hook = hook
        self.hook_list = hook_list

    def remove(self, absent_ok: bool = True):
        try:
            idx = self.hook_list.index(self.hook)
        except ValueError as e:
            if absent_ok:
                return
            else:
                raise e
        self.hook_list.pop(idx)


class VarHookMixIn:  # to be documented
    _capture_hooks: dict[str, list[VarHook]]

    def register_variable_hook(self, name: str, hook: VarHook) -> HookRemover:
        hooks = self._get_capture_hooks(True)
        if name in hooks:
            hooks[name].append(hook)
        else:
            hooks[name] = [hook]
        return HookRemover(hook, hooks[name])

    def remove_variable_hook(self, name: str):
        hooks = self._get_capture_hooks()
        if hooks is not None:
            hooks.pop(name, None)  # give default to avoid KeyError

    def variable_hook(self, name: str, obj: _T) -> _T:
        hooks = self._get_capture_hooks()
        if hooks is None:
            return obj

        hook_list = hooks.get(name, None)
        # if hook is None:
        #     hook = hooks.get('*', None)
        if hook_list is not None:
            for hook in hook_list:
                ret = hook(obj)
                if ret is not None:
                    obj = ret
        return obj

    def hook_registered(self, name: str) -> bool:
        hooks = self._get_capture_hooks()
        if hooks is None:
            return False
        return name in hooks

    def _get_capture_hooks(self, create: bool = False) -> dict[str, list[VarHook]]:
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


Cond = typing.Callable[[typing.Numeric], bool | torch.BoolTensor]
Expr = typing.Callable[[typing.Numeric], typing.Any]
Exprs = Expr | typing.Sequence[Expr]


class Conditional:

    def __init__(
        self,
        condition: Cond,
        expr_true: Exprs,
        expr_false: Exprs,
        condition_tensor: Cond = None,
        expr_true_tensor: Exprs = None,
        expr_false_tensor: Exprs = None,
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
                return self._eval(self.expr_true_tensor, value)
            else:
                result = self._eval(self.expr_false_tensor, value)
                if condition.any():
                    result_true = self._eval(self.expr_true_tensor, value)
                    true_atomic = not isinstance(self.expr_true_tensor, collections.abc.Sequence)
                    false_atomic = not isinstance(self.expr_false_tensor, collections.abc.Sequence)
                    if true_atomic and false_atomic:
                        return torch.where(condition, result_true, result)
                    else:
                        if true_atomic:
                            result_true = [result_true for _ in range(len(result))]  # result is a list
                        if false_atomic:
                            result = [result for _ in range(len(result_true))]  # result_true is a list
                        return [torch.where(condition, r1, r2) for r1, r2 in zip(result_true, result)]
                return result
        else:
            if self.condition(value):
                return self.expr_true(value)
            else:
                return self.expr_false(value)

    @staticmethod
    def _eval(expr, value):
        if isinstance(expr, collections.abc.Sequence):
            return [exp(value) for exp in expr]
        else:
            return expr(value)


class InfinityCond(Conditional):
    def __init__(
        self,
        expr_finite: Exprs,
        expr_infinite: Exprs,
        expr_finite_tensor: Exprs = None,
        expr_infinite_tensor: Exprs = None
    ):
        super().__init__(
            lambda x: x == float('inf'), expr_infinite, expr_finite,
            lambda x: x.isinf(), expr_infinite_tensor, expr_finite_tensor
        )


class GenericCompute:
    zero: 'GenericCompute'
    one: 'GenericCompute'
    cos: 'GenericCompute'
    sin: 'GenericCompute'
    tan: 'GenericCompute'
    acos: 'GenericCompute'
    asin: 'GenericCompute'
    exp: 'GenericCompute'
    log: 'GenericCompute'
    sqrt: 'GenericCompute'
    abs: 'GenericCompute'

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


# @formatter:off
GenericCompute.zero = GenericCompute(lambda x: 0 * x, torch.zeros_like)
GenericCompute.one  = GenericCompute(lambda x: 1 if isinstance(x, int) else 1., torch.ones_like)
GenericCompute.cos  = GenericCompute(math.cos, torch.cos)
GenericCompute.sin  = GenericCompute(math.sin, torch.sin)
GenericCompute.tan  = GenericCompute(math.tan, torch.tan)
GenericCompute.acos = GenericCompute(math.acos, torch.acos)
GenericCompute.asin = GenericCompute(math.asin, torch.asin)
GenericCompute.exp  = GenericCompute(math.exp, torch.exp)
GenericCompute.log  = GenericCompute(math.log, torch.log)
GenericCompute.sqrt = GenericCompute(math.sqrt, torch.sqrt)
GenericCompute.abs  = GenericCompute(math.fabs, torch.abs)
# @formatter:on


class ContextCache:
    """
    A class to enable cacheing return values of its methods.
    First, decorate the methods that may be cached with :func:`context_cache`.
    Then, start a "with" block with :meth:`enable_cache` and the specified methods
    are cached in the block. At the end of the block, the cache is cleared.

    .. warning::
        Specified methods are cached **unconditionally**, i.e. the second call
        returns what the first call returns exactly.
    """
    _ctx_cache: dict

    @contextlib.contextmanager
    def enable_cache(self, items: str | typing.Collection[str]):
        """
        Return a context manager that caches the specified methods.

        :param items: A collection of keys of cached methods.
        """
        if isinstance(items, str):
            items = [items]

        original = self._get_ctx_cache()
        cache = {}
        for item in items:
            cache[item] = original.get(item, None)
        self._ctx_cache = cache

        yield cache

        self._ctx_cache = original

    def _get_ctx_cache(self):
        d = self.__dict__
        d.setdefault('_ctx_cache', {})
        return d['_ctx_cache']


def context_cache(func: typing.Callable | str = None):
    """
    A decorator to mark a method as cacheable by :class:`ContextCache`.
    If called with one argument of type ``str``, it serves as the key of the method.
    If called with no argument, the key is the name of the method.
    """
    if callable(func):
        f_name = func.__name__
        return context_cache(f_name)(func)

    def decorator(f):
        cache_key = func
        if cache_key is None:
            cache_key = f.__name__

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            cache = getattr(self, '_ctx_cache', {})
            if cache_key not in cache:
                return f(self, *args, **kwargs)

            if cache[cache_key] is None:
                cache[cache_key] = f(self, *args, **kwargs)
            return cache[cache_key]

        return wrapper

    return decorator


@contextlib.contextmanager
def enable_group_cache(items: str | typing.Collection[str], objs: list):
    with contextlib.ExitStack() as stack:
        entries = [stack.enter_context(obj.enable_cache(items)) for obj in objs]
        yield entries
