import torch

from .. import base
from ..base.typing import Ts, Literal, Self

__all__ = [
    'norm_psf',

    'PsfRecenterType',

    'MeanPsfRecenter',
    'NullPsfRecenter',
    'PsfRecenter',
    'SingleWlPsfRecenter',
    'WlWisePsfRecenter',
]


def norm_psf(psf: Ts, dims: tuple[int, int] = (-2, -1)) -> Ts:
    r"""
    Normalizes PSF so that all its pixels sum up to 1.

    :param Tensor psf: PSF to normalize. It cannot be complex of have negative elements.
    :param dims: Indices of spatial dimensions of ``psf``. Default: ``(-2, -1)``.
    :type dims: tuple[int, int]
    :return: Normalized PSF.
    :rtype: Tensor
    """
    if torch.is_complex(psf):
        raise ValueError(f'Complex PSF cannot be normalized')
    if psf.lt(0).any():
        raise ValueError(f'PSF with negative values cannot be normalized')
    den = psf.sum(dims, True)
    psf = torch.where(den.ne(0), psf / den, 0)
    return psf


PsfRecenterType = Literal['', 'none', 'center', 'mean']


class PsfRecenter:
    def __call__(self, psf: Ts) -> Ts:
        shape = psf.shape
        if len(shape) < 3:
            raise base.ShapeError(f'PSF must have at least 3 dimensions, got shape {shape}')

        if len(shape) == 3:
            psf = psf.unsqueeze(0)  # (1,N_wl,H,W)
        psf = psf.flatten(0, -4)  # (N,N_wl,H,W)

        values, indices_r = psf.max(-2)  # (N,N_wl,W)
        _, index_c = values.max(-1)  # (N,N_wl)
        index_r = torch.gather(indices_r, 2, index_c.unsqueeze(-1)).squeeze(-1)  # (N,N_wl)

        center_r, center_c = shape[-2] // 2, shape[-1] // 2
        shift_r, shift_c = center_r - index_r, center_c - index_c
        shift_r, shift_c = self.shift_reduce(psf, shift_r, shift_c)
        shift_shape = psf.shape[:2]
        shift_r, shift_c = shift_r.broadcast_to(shift_shape), shift_c.broadcast_to(shift_shape)

        psf = psf.flatten(0, 1)  # (N*N_wl,H,W)
        shift_r = shift_r.flatten()
        shift_c = shift_c.flatten()
        psf = torch.stack([
            torch.roll(psf[i], (shift_r[i].item(), shift_c[i].item()), (-2, -1))
            for i in range(psf.size(0))
        ])

        psf = psf.reshape(shape)
        return psf

    @torch.no_grad()
    def shift_reduce(self, psf: Ts, shift_r: Ts, shift_c: Ts) -> tuple[Ts, Ts]:
        raise NotImplementedError()

    @classmethod
    def create(cls, type_: PsfRecenterType | bool | int = False) -> Self:
        if isinstance(type_, bool):
            return WlWisePsfRecenter() if type_ else NullPsfRecenter()
        elif isinstance(type_, int):
            return SingleWlPsfRecenter(type_)
        elif isinstance(type_, str):
            if type_ == 'none':
                return WlWisePsfRecenter()
            elif type_ == 'center':
                return SingleWlPsfRecenter()
            elif type_ == 'mean':
                return MeanPsfRecenter()
            else:
                raise ValueError(f'Unknown PSF recenter type: {type_}')
        raise ValueError(f'Unknown PSF recenter type: {type_}')


class NullPsfRecenter(PsfRecenter):
    def __call__(self, psf: Ts) -> Ts:
        return psf


class WlWisePsfRecenter(PsfRecenter):

    def shift_reduce(self, psf: Ts, shift_r: Ts, shift_c: Ts) -> tuple[Ts, Ts]:
        return shift_r, shift_c  # do not reduce


class SingleWlPsfRecenter(PsfRecenter):
    def __init__(self, wl_index: int = None):
        self.wl_index = wl_index

    def shift_reduce(self, psf: Ts, shift_r: Ts, shift_c: Ts) -> tuple[Ts, Ts]:
        idx = self.wl_index
        if idx is None:
            idx = psf.shape[1] // 2  # center wavelength
        shift_r = shift_r[:, [idx]]
        shift_c = shift_c[:, [idx]]
        return shift_r, shift_c


class MeanPsfRecenter(PsfRecenter):
    def shift_reduce(self, psf: Ts, shift_r: Ts, shift_c: Ts) -> tuple[Ts, Ts]:
        shift_r = shift_r.mean(0).unsqueeze(0)
        shift_c = shift_c.mean(0).unsqueeze(0)
        return shift_r, shift_c
