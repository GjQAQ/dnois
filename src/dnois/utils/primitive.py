import inspect
import typing
import warnings

__all__ = [
    'get_bound_args',
    'subclasses',
]

_T = typing.TypeVar('_T')


def _subclasses(cls: type[_T], no_abstract: bool, no_protected: bool) -> set[type[_T]]:
    def ff(clz):
        if not inspect.isclass(clz):
            return False
        if no_abstract and inspect.isabstract(clz):
            return False
        if no_protected and clz.__name__.startswith('_'):
            return False
        return True

    subs = cls.__subclasses__()
    required = set(filter(ff, subs))  # use set to avoid duplicates
    for sub in subs:
        required = required | _subclasses(sub, no_abstract, no_protected)
    return required


def subclasses(cls: type[_T], no_abstract: bool = True, no_protected: bool = True) -> list[type[_T]]:
    """
    Returns a list of subclasses of the given class, sorted by their names.

    :param type cls: The class to find subclasses of.
    :param bool no_abstract: If ``True``, abstract classes are not
        included in the result. Default: ``True``.
    :param bool no_protected: If ``True``, classes with names starting
        with '_' are not included in the result. Default: ``True``.
    :return: A list of subclasses of the given class, sorted by their names.
    :rtype: list[type]
    """
    sub_list = _subclasses(cls, no_abstract, no_protected)
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
