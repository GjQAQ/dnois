import functools
import inspect

from .. import base
from ..base import typing

__all__ = [
    'fmt',
    'subclasses',
    'with_external',

    'VarCollection',
    'VarDict',
    'VarHook',
    'VarHookMixIn',
    'ExternalParamMixIn',
    'FixStateMixIn',
]


def _subclasses(cls: type) -> set[type]:
    subs = set(cls.__subclasses__())  # use set to avoid duplicates
    for sub in subs.copy():
        subs = subs | _subclasses(sub)
    return subs


def subclasses(cls: type, _filter: bool = True) -> list[type]:
    # Returns subclasses of cls recursively
    # If _filter is True, only non-abstract and non-private (name starting with _) classes are returned
    sub_list = _subclasses(cls)
    if _filter:
        sub_list = list(filter(lambda c: not inspect.isabstract(c) and not c.__name__.startswith('_'), sub_list))
    sub_list = sorted(sub_list, key=lambda c: c.__name__)
    return sub_list


def fmt(v: float) -> str:
    """
    Format a ``float`` according to :data:`dnois.float_print_fmt`.

    :param float v: The ``float`` to be formatted.
    :return: Formatted string.
    :rtype: str
    """
    s = f'{{:{base.float_print_fmt}}}'
    return s.format(v)


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


VarCollection=VarDict
