import torch
import warnings

from .. import utils, fourier, base, torch as _t
from ..base.typing import Ts, Size2d, size2d, cast

__all__ = [
    'depth_aware',
    'simple',
    'space_variant',
    'spots2image',
    'superpose',

    'DepthAware',
    'PSFAugSimple',
    'Simple',
    'SpaceVariant',
]


def simple(obj: Ts, psf: Ts, pad: Size2d | str = 'linear', compensate_edge: bool = False, eps: float = 1e-3) -> Ts:
    r"""
    Simplest image formation model. Blurred image is computed by a convolution between the
    object (sharp image) and PSF, implemented by FFT.

    Note that both ``obj`` and ``psf`` can be real representing incoherent imaging and
    complex representing coherent imaging. If just one of them is complex, the other will
    be cast as complex. The blurred image is real if and only if they are both real.

    If linear convolution is computed, the edge of blurred image will fade out. If ``compensate_edge``
    is set to ``True``, it will be further divided by convolution between PSF and a totally white
    image to compensate this issue. In this case ``eps`` is added to the denominator to avoid division by zero.

    :param Tensor obj: A tensor of shape :math:`(\cdots,H_o,W_o)`.
    :param Tensor psf: A tensor of shape :math:`(\cdots,H_p,W_p)`.
    :param pad: Padding width used to mitigate aliasing. See :func:`dnois.fourier.dconv2`
        for more details. Default: ``'linear'``.
    :type pad: int, tuple[int, int] or str
    :param bool compensate_edge: Whether to compensate the edge fading out,
        see description above. Default: ``False``.
    :param float eps: A small positive number to avoid division by zero,
        see description above. Default: ``1e-3``.
    :return: A tensor of shape :math:`(\cdots,H_o,W_o)`.
    :rtype: Tensor
    """
    if pad != 'linear' and compensate_edge:
        warnings.warn(f'{compensate_edge=} is ignored when {pad=}')

    blurred = fourier.dconv2(obj, psf, out='same', padding=pad)
    blurred = utils.resize(blurred, obj.shape[-2:])  # in case PSF is larger

    if compensate_edge:
        msk = torch.ones_like(obj)
        msk = fourier.dconv2(msk, psf, out='same', padding=pad)
        msk = utils.resize(msk, obj.shape[-2:])
        blurred = blurred / (msk + eps)
    return blurred


def depth_aware(
    obj: Ts,
    mask: Ts,
    psf: Ts,
    pad: Size2d | str = 'linear',
    occlusion_aware: bool = False,
) -> Ts:
    r"""
    Depth-aware image formation model. The ``obj`` (sharp image) is first segmented into
    depth planes w.r.t. depth according to ``mask``. Then the portion in each plane is
    convolved with the PSF of corresponding depth. Final image (blurred image) is the
    superposition of all the planes.

    Note that both ``obj`` and ``psf`` can be real representing incoherent imaging and
    complex representing coherent imaging. If just one of them is complex, the other will
    be cast as complex. The blurred image is real if and only if they are both real.

    Smaller indices in :math:`D` dimension in ``mask`` and ``psf`` are expected to
    represent smaller depths.

    :param Tensor obj: A tensor of shape :math:`(\cdots,H_o,W_o)`.
    :param Tensor mask: A tensor of shape :math:`(\cdots,D,H_o,W_o)`.
    :param Tensor psf: A tensor of shape :math:`(\cdots,D,H_p,W_p)`. PSF should be normalized,
        see :func:`dnois.optics.norm_psf` for details.
    :param pad: Padding width used to mitigate aliasing. See :func:`dnois.fourier.dconv2`
        for more details. Default: ``'linear'``.
    :type pad: int, tuple[int, int] or str
    :param bool occlusion_aware: Whether to use the occlusion-aware image formation algorithm
        [#occ]_. Default: ``False``.
    :return: A tensor of shape :math:`(\cdots,H_o,W_o)`.
    :rtype: Tensor

    .. [#occ] Ikoma, H., Nguyen, C. M., Metzler, C. A., Peng, Y., & Wetzstein, G. (2021, May).
        Depth from defocus with learned optics for imaging and occlusion-aware depth estimation.
        In 2021 IEEE International Conference on Computational Photography (ICCP) (pp. 1-12). IEEE.
    """
    if psf.size(-2) > obj.size(-2) or psf.size(-1) > obj.size(-1):
        raise base.ShapeError(f'Spatial dimension of PSF ({psf.shape[-2:]}) cannot '
                              f'be larger than that of object ({obj.shape[-2:]})')
    if mask.gt(1).any() or mask.lt(0).any():
        raise ValueError(f'Value of mask must lie in [0, 1]')

    slices = obj.unsqueeze(-3) * mask  # ... x D x H x W
    if occlusion_aware:
        accum = torch.flip(torch.cumsum(torch.flip(mask, (-3,)), -3), (-3,))
        blr_accum, blr_mask, blr_img = fourier.dconv_mult([accum, mask, slices], psf, (-2, -1), 'same', pad)
        blr_accum = blr_accum.clamp(min=1e-5)
        blr_mask = blr_mask / blr_accum

        acc_prod = torch.cumprod(1 - blr_mask, -3)
        acc_prod = torch.roll(acc_prod, 1, -3)
        acc_prod[..., 0, :, :] = 1

        blr_img = blr_img / blr_accum
        blurred = torch.sum(acc_prod * blr_img, dim=-3)
    else:
        blurred_slices = fourier.dconv2(slices, psf, out='same', padding=pad)  # ... x D x H x W
        blurred = blurred_slices.sum(dim=-3)  # ... x H x W
    return blurred


def superpose(obj: Ts, psf: Ts) -> Ts:
    r"""
    Point-wise image formation model. The image is the superposition of PSFs
    of all object points.

    Note that both ``obj`` and ``psf`` can be real representing incoherent imaging and
    complex representing coherent imaging. If just one of them is complex, the other will
    be cast as complex. The blurred image is real if and only if they are both real.

    :param Tensor obj: A tensor of shape :math:`(\cdots,H_o,W_o)`.
    :param Tensor psf: A tensor of shape :math:`(\cdots,H_o,W_o,H_p,W_p)`.
    :return: A tensor of shape :math:`(\cdots,H_o,W_o)`.
    :rtype: Tensor
    """
    image = torch.zeros_like(obj)
    for i in range(obj.size(-2)):
        for j in range(obj.size(-1)):
            upper1 = max(0, i - psf.size(-2) // 2)
            lower1 = min(obj.size(-2) - 1, i + (psf.size(-2) - 1) // 2)
            left1 = max(0, j - psf.size(-1) // 2)
            right1 = min(obj.size(-1) - 1, j + (psf.size(-1) - 1) // 2)
            upper2 = max(0, psf.size(-2) // 2 - i)
            left2 = max(0, psf.size(-1) // 2 - j)
            image[..., upper1:lower1 + 1, left1:right1 + 1] += (
                obj[..., i, j] * psf[..., i, j, upper2 + (lower1 - upper1), left2 + (right1 - left1)]
            )
    return image


def space_variant(
    obj: Ts,
    psf: Ts,
    pad: Size2d = 0,
    linear_conv: bool = False,
    _one_by_one: bool = False,
) -> Ts:
    r"""
    Space-variant image formation model. The image plane is partitioned into non-overlapping
    patches. The PSF in each patch is assumed to be space-invariant and convolved with the
    corresponding patch to obtain the blurred patch. Number of patches is determined by the
    dimensionality of ``psf``. This is typically used to blur an image with FoV-dependent PSF.

    To mitigate the abrupt change between PSFs in different patches, each patch can be
    padded with pixels from neighboring patches where padding amount is specified by ``pad``.
    Adjacent patches overlap after padding and overlapping regions are merged.
    Note that merging method must be ``crop`` if ``linear_conv`` is ``False``.

    Note that both ``obj`` and ``psf`` can be real representing incoherent imaging and
    complex representing coherent imaging. If just one of them is complex, the other will
    be cast as complex. The blurred image is real if and only if they are both real.

    :param Tensor obj: A tensor of shape :math:`(\cdots,H_o,W_o)`.
    :param Tensor psf: A tensor of shape :math:`(\cdots,N_h,N_w,H_p,W_p)`.
    :param pad: Padding width in vertical and horizontal directions. Default: 0.
    :type pad: int | tuple[int, int]
    :param bool linear_conv: Whether to use linear convolution in each patch. Default: False.
    :return: A tensor of shape :math:`(\cdots,H_o,W_o)`.
    :rtype: Tensor
    """
    pad = size2d(pad)
    patches = utils.partition_padded(obj, psf.shape[-4:-2], pad, 'replicate')
    # ... x N_h x N_w x H_p x W_p
    patches = torch.stack([torch.stack(cast(list[Ts], row), -3) for row in patches], -4)

    wh = torch.linspace(0, 1, pad[0] * 2 + 2, device=obj.device, dtype=obj.dtype)[1:-1, None]
    ww = torch.linspace(0, 1, pad[1] * 2 + 2, device=obj.device, dtype=obj.dtype)[None, 1:-1]
    if pad[0] != 0:
        patches[..., 1:, :, :pad[0] * 2, :] *= wh
        patches[..., :-1, :, -pad[0] * 2:, :] *= wh.flip(0)
    if pad[1] != 0:
        patches[..., :, 1:, :, :pad[1] * 2] *= ww
        patches[..., :, :-1, :, -pad[1] * 2:] *= ww.flip(1)

    if linear_conv:
        padding = 'linear'
        overlap = (psf.size(-2) - 1 + 2 * pad[0], psf.size(-1) - 1 + 2 * pad[1])
    else:
        padding = 'none'
        overlap = (2 * pad[0], 2 * pad[1])

    if _one_by_one:
        blurred = torch.stack([
            torch.stack([
                fourier.dconv2(psf_elem, patch, out='full', padding=padding)
                for psf_elem, patch in zip(psf_row.unbind(-3), patch_row.unbind(-3))
            ], -3) for psf_row, patch_row in zip(psf.unbind(-4), patches.unbind(-4))
        ], -4)
    else:
        # ... x N_h x N_w x H_p x W_p
        blurred = fourier.dconv2(psf, patches, out='full', padding=padding)

    blurred = blurred.transpose(-4, 0).transpose(-3, 1)  # N_h x N_w x ... x H_p x W_p
    blurred = utils.merge_patches(blurred, overlap, 'sum')
    blurred = utils.crop(blurred, pad)
    if linear_conv:
        blurred = blurred.narrow(-2, psf.size(-2) // 2, obj.size(-2))
        blurred = blurred.narrow(-1, psf.size(-1) // 2, obj.size(-1))

    return blurred


def spots2image(size: Size2d, r: Ts, c: Ts, value: Ts, mask: Ts = None):
    """
    Superpose ``N`` spots to form an image. For example, in ray tracing some rays intersect with
    an image plane and their pattern reflects the image they form.

    .. note::
        The index of spots can be floating point numbers.

    :param size: Size of the image in pixels ``(H, W)``.
    :type size: int | tuple[int, int]
    :param Tensor r: Row indices of the spots. A tensor of shape ``(..., N, spp)``.
    :param Tensor c: Column indices of the spots. A tensor of shape ``(..., N, spp)``.
    :param Tensor value: Values of the spots. A tensor of shape ``(..., N, spp)``.
    :param Tensor mask: A validity mask with same shape as ``r`` and ``c``.
        If not given, all spots are considered valid.
    :return: Formed image. A tensor of shape ``(..., H, W)``.
    :rtype: Tensor
    """
    size = size2d(size)
    if mask is None:
        broadcastable = _t.broadcastable(r, c, value)
        pre_shape = torch.broadcast_shapes(r.shape, c.shape, value.shape)[:-2]
        _1, _2 = 'r and c', f'{r.shape}, {c.shape} and {value.shape}'
    else:
        broadcastable = _t.broadcastable(r, c, mask, value)
        pre_shape = torch.broadcast_shapes(r.shape, c.shape, mask.shape, value.shape)[:-2]
        _1, _2 = 'r, c and mask', f'{r.shape}, {c.shape}, {mask.shape} and {value.shape}'
    if not broadcastable:
        raise _t.ShapeError(f'{_1} must be broadcastable and their shape except the last dimension '
                            f'must be broadcastable with value, but got {_2}')

    h, w = size
    c_a, r_a = torch.floor(c.detach() + 0.5).long(), torch.floor(r.detach() + 0.5).long()
    in_region = (c_a >= 0) & (c_a <= w) & (r_a >= 0) & (r_a <= h)  # (..., N, spp)
    c_a[~in_region] = 0
    r_a[~in_region] = 0
    c_as, r_as = c_a - 1, r_a - 1
    w_c, w_r = c_a - c + 0.5, r_a - r + 0.5
    iw_c, iw_r = 1 - w_c, 1 - w_r
    if mask is None:
        mask = in_region
    else:
        mask = mask & in_region

    pre_idx = [
        _t.as1d(torch.arange(dim_size, device=value.device), len(pre_shape) + 2, i)
        for i, dim_size in enumerate(pre_shape)
    ]  # (..., N, spp)
    image = value.new_zeros(pre_shape + (h + 2, w + 2))  # (..., H+2, W+2)
    v1 = value * w_c  # (..., N, spp)
    v2 = value * iw_c
    image.index_put_(pre_idx + [r_as, c_as], torch.where(mask, v1 * w_r, 0), True)  # top left
    image.index_put_(pre_idx + [r_a, c_as], torch.where(mask, v1 * iw_r, 0), True)  # bottom left
    image.index_put_(pre_idx + [r_as, c_a], torch.where(mask, v2 * w_r, 0), True)  # top right
    image.index_put_(pre_idx + [r_a, c_a], torch.where(mask, v2 * iw_r, 0), True)  # bottom right
    image = image[..., :-2, :-2] / mask.size(-1)
    return image


class Simple(torch.nn.Module):
    """Module wrapper for :func:`simple`."""

    def __init__(self, pad: Size2d | str = 'linear', compensate_edge: bool = False, eps: float = 1e-3):
        super().__init__()
        self.pad = pad
        self.compensate_edge = compensate_edge
        self.eps = eps

    def forward(self, psf: Ts, obj: Ts) -> Ts:
        """See :func:`simple`."""
        return simple(obj, psf, self.pad, self.compensate_edge, self.eps)


class PSFAugSimple(Simple):
    def __init__(self, aug_type: str = 'flip', **kwargs):
        super().__init__(**kwargs)
        self.aug_type = aug_type

    def forward(self, psf: Ts, obj: Ts) -> Ts:
        psf = psf.squeeze()
        if psf.ndim != 3:
            raise RuntimeError(f'Unexpected PSF shape {psf.shape} for {self.__class__.__name__}')
        if obj.ndim != 4:
            raise RuntimeError(f'Image of shape (B,C,H,W) expected, got {psf.shape}')

        if self.aug_type == 'flip':
            b = obj.size(0)
            single_size, r = divmod(b, 4)
            psf = psf.unsqueeze(0)
            psf = torch.stack([
                psf.expand(single_size + int(r >= 1), -1, -1, -1),
                psf.fliplr().expand(single_size + int(r >= 2), -1, -1, -1),
                psf.flipud().expand(single_size + int(r >= 3), -1, -1, -1),
                psf.flip(-2, -1).expand(single_size, -1, -1, -1),
            ])
            return super().forward(psf, obj)
        else:
            raise RuntimeError(f'Unknown aug_type: {self.aug_type}')


class DepthAware(torch.nn.Module):
    """Module wrapper for :func:`depth_aware`."""

    def __init__(self, pad: Size2d | str = 'linear', occlusion_aware: bool = False):
        super().__init__()
        self.pad = pad
        self.occlusion_aware = occlusion_aware

    def forward(self, psf: Ts, obj: Ts, mask: Ts) -> Ts:
        """See :func:`depth_aware`."""
        return depth_aware(obj, mask, psf, self.pad, self.occlusion_aware)


class SpaceVariant(torch.nn.Module):
    """Module wrapper for :func:`space_variant`."""

    def __init__(self, pad: Size2d = 0, linear_conv: bool = False, _one_by_one: bool = False):
        super().__init__()
        self.pad = pad
        self.linear_conv = linear_conv
        self._one_by_one = _one_by_one

    def forward(self, psf: Ts, obj: Ts) -> Ts:
        """See :func:`space_variant`."""
        return space_variant(obj, psf, self.pad, self.linear_conv, self._one_by_one)
