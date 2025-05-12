import inspect
import warnings

from ..base import typing

__all__ = [
    'get_bound_args',
    'subclasses',
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


_empty = inspect.Parameter.empty


def _match_annotation(ba: inspect.BoundArguments, params) -> bool:
    for name, value in ba.arguments.items():
        param: inspect.Parameter = params[name]
        annt = param.annotation
        if annt is not _empty and not isinstance(value, annt):
            return False
    return True


def get_bound_args(func, match_annotation, *args, **kwargs) -> inspect.BoundArguments:
    ols = typing.get_overloads(func)
    if not ols:
        warnings.warn(f'Trying to {get_bound_args.__name__} on a function without overloads')
        return inspect.signature(func).bind(*args, **kwargs)

    self = getattr(func, '__self__', None)
    if self is not None:
        args = (self,) + args
    for ol in ols:
        sig = inspect.signature(ol)
        try:
            ba = sig.bind(*args, **kwargs)
        except TypeError:
            continue
        else:
            if match_annotation and _match_annotation(ba, sig.parameters):
                return ba
    raise TypeError(f'Cannot find a valid overload of {func.__name__} to bind arguments to')
