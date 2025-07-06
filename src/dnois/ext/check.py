import functools

__all__ = [
    'requires',
]


def requires(*deps):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for dep in deps:
                try:
                    __import__(dep)
                except ImportError:
                    raise ImportError(f'Package {dep} is required to use {func.__qualname__}')
            return func(*args, **kwargs)

        return wrapper

    return decorator
