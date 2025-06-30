"""
``dnois.conf`` defines some constants used globally in DNOIS.
They can be modified to change the behavior of DNOIS.
"""
import functools

#: Default format to print float numbers.
float_print_fmt: str = '.6g'
#: Expansion percentage of aperture radius when determining its passing region.
detection_radius_eps: float = 1e-5

_config_items = [k for k in globals().keys() if not k.startswith('_')]


class config:  # noqa
    """
    A context manager to temporarily change global configurations.

    >>> import dnois
    >>> with dnois.config(float_print_fmt='.3f'):
    ...     print(dnois.fmt(1.23456789))
    1.235
    >>> print(dnois.fmt(1.23456789))
    1.23457

    .. note::
        The usage of this class is similar to ``torch.no_grad()``,
        which can serve as a context manager or a decorator.

    .. warning::
        The temporary changes made by this context manager is not thread-safe.

    :param configs: Keyword arguments to change global configurations.
        Each key must be a valid configuration item name.
    """

    def __init__(self, **configs):
        for k in configs.keys():
            if k not in _config_items:
                raise ValueError(f'Unknown config item: {k}')
        self._configs = configs

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)

        return wrapper

    def __enter__(self):
        self._old_configs = {k: globals()[k] for k in self._configs.keys()}
        for k, v in self._configs.items():
            globals()[k] = v

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k, v in self._old_configs.items():
            globals()[k] = v
