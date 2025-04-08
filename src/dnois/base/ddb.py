"""DNOIS debugging utilities."""
from .typing import Ts, Callable

__all__ = [
    'debug',
    'debugging',
]
_debug = False


def debug(on: bool = True):
    """
    Switch :doc:`debugging mode </content/guide/debug>` on or off.

    :param bool on: Enable debugging mode if ``True``, disable it otherwise. Default: ``True``.
    :return: ``None``
    """
    global _debug
    _debug = bool(on)


def debugging() -> bool:
    """
    Returns whether debugging is enabled or not.

    :return: Whether debugging is enabled or not.
    :rtype: bool
    """
    return _debug


def grad_hook_check_peculiar(name: str, msg: str = None) -> Callable[[Ts], None]:
    """
    Returns a tensor backward hook function (see :meth:`torch.Tensor.register_hook`)
    to check whether the gradient w.r.t. to the hooked tensor contains ``inf``, ``-inf``
    or ``nan``. A :class:`RuntimeError` will be raised if so.

    :param str name: The name of hooked tensor for error message.
    :param str msg: Additional information appended to the error message. Default: do not append.
    :return: The hook function.
    :rtype: Callable[[Tensor], None]
    """

    def _check_peculiar(grad: Ts):
        n_nan = grad.isnan().sum()
        n_inf = grad.isinf().sum()
        if n_nan > 0 or n_inf > 0:
            total = grad.numel()
            text = (f'Grad w.r.t. {name} contains peculiar values: '
                    f'{n_nan / total:.2%} NaN and {n_inf / total * 100:.2f}% Inf.')
            if msg is not None:
                text += f' Complement: {msg}'
            raise RuntimeError(text)

    return _check_peculiar


def check_peculiar(x: Ts, name: str, msg: str = None):
    if x.requires_grad:
        x.register_hook(grad_hook_check_peculiar(name, msg))
