import math

import torch

from ..base.typing import Numeric, Ts, overload

__all__ = [
    'circle_of_confusion',
    'imgd',
    'norm_psf',
    'objd',
]


@overload
def circle_of_confusion(pupil_diameter: Numeric, fl: Numeric, d: Numeric, focal_d: Numeric) -> Numeric:
    pass


@overload
def circle_of_confusion(pupil_diameter: Numeric, sensor_distance: Numeric, image_distance: Numeric) -> Numeric:
    pass


def circle_of_confusion(pupil_diameter: Numeric, fl: Numeric, d: Numeric, focal_d: Numeric = None) -> Numeric:
    r"""
    Returns the diameter of circle of confusion :math:`D_\text{COC}`.

    Thin function has two overloaded forms:

    .. function:: circle_of_confusion(pupil_diameter, fl, d, focal_d)
        :no-index:

        .. math::
            D_\text{COC}=D_\text{pupil}\frac{f(d_\text{f}-d)}{d(d_\text{f}-f)}

        :param pupil_diameter: Diameter of the pupil :math:`D_\text{pupil}`.
        :param fl: Focal length :math:`f`.
        :param d: Object distance :math:`d`.
        :param focal_d: Focal object distance :math:`d_\text{f}`.

    .. function:: circle_of_confusion(pupil_diameter, sensor_distance, image_distance)
        :no-index:

        .. math::
            D_\text{COC}=D_\text{pupil}\left(1-\frac{d_\text{s}}{d_\text{i}}\right)

        :param pupil_diameter: Diameter of the pupil :math:`D_\text{pupil}`.
        :param sensor_distance: Distance between lens and image plane :math:`d_\text{s}`.
        :param image_distance: Image distance :math:`d_\text{i}`.

    :return: Diameter of circle of confusion :math:`D_\text{COC}`.
    """
    if focal_d is None:
        sensor_distance = fl
        image_distance = d
        factor = 1 - sensor_distance / image_distance
    else:
        factor = (fl * (focal_d - d)) / (d * (focal_d - fl))
    coc = pupil_diameter * factor

    if torch.is_tensor(coc):
        return coc.abs()
    else:
        return math.fabs(coc)


@overload
def objd(img_d: Numeric, fl_obj: Numeric, fl_img: Numeric = None) -> Numeric:
    pass


@overload
def objd(img_d: Numeric, n_obj: Numeric, n_img: Numeric, diopter: Numeric) -> Numeric:
    pass


def objd(img_d: Numeric, fl_obj: Numeric, fl_img: Numeric = None, diopter: Numeric = None) -> Numeric:
    r"""
    Returns object distance :math:`s` given image distance :math:`s'`.

    This function has two overloaded forms:

    .. function:: objd(img_d, fl_obj, fl_img = None)
        :no-index:

        .. math::
            s=\frac{fs'}{s'-f'}

        :param img_d: Image distance :math:`s'`.
        :type img_d: float or Tensor
        :param fl_obj: Object focal length :math:`f`.
        :type fl_obj: float or Tensor
        :param fl_img: Image focal length :math:`f'`. Default: identical to ``fl_obj``.
        :type fl_img: float or Tensor

    .. function:: objd(img_d, n_obj, n_img, diopter)
        :no-index:

        .. math::
            s=\frac{n_1s'}{\phi s'-n_2}

        :param img_d: Image distance :math:`s'`.
        :type img_d: float or Tensor
        :param n_obj: Refractive index in object space :math:`n_1`.
        :type n_obj: float or Tensor
        :param n_img: Refractive index in image space :math:`n_2`.
        :type n_img: float or Tensor
        :param diopter: Diopter :math:`\phi`.
        :type diopter: float or Tensor

    :return: Object distance :math:`s`.
    :rtype: float or Tensor
    """
    if diopter is None:
        if fl_img is None:
            fl_img = fl_obj
        return fl_obj / (1 - fl_img / img_d)
    else:
        n_obj, n_img = fl_obj, fl_img
        return n_obj * img_d / (diopter * img_d - n_img)


@overload
def imgd(obj_d: Numeric, fl_obj: Numeric, fl_img: Numeric = None) -> Numeric:
    pass


@overload
def imgd(obj_d: Numeric, n_obj: Numeric, n_img: Numeric, diopter: Numeric) -> Numeric:
    pass


def imgd(obj_d: Numeric, fl_obj: Numeric, fl_img: Numeric = None, diopter: Numeric = None) -> Numeric:
    r"""
    Returns image distance :math:`s'` given object distance :math:`s`.

    This function has two overloaded forms:

    .. function:: imgd(obj_d, fl_obj, fl_img = None)
        :no-index:

        .. math::
            s'=\frac{f's}{s-f}

        :param obj_d: Object distance :math:`s`.
        :type obj_d: float or Tensor
        :param fl_obj: Object focal length :math:`f`.
        :type fl_obj: float or Tensor
        :param fl_img: Image focal length :math:`f'`. Default: identical to ``fl_obj``.
        :type fl_img: float or Tensor

    .. function:: imgd(obj_d, n_obj, n_img, diopter)
        :no-index:

        .. math::
            s'=\frac{n_2s}{\phi s-n_1}

        :param obj_d: Object distance :math:`s`.
        :type obj_d: float or Tensor
        :param n_obj: Refractive index in object space :math:`n_1`.
        :type n_obj: float or Tensor
        :param n_img: Refractive index in image space :math:`n_2`.
        :type n_img: float or Tensor
        :param diopter: Diopter :math:`\phi`.
        :type diopter: float or Tensor

    :return: Image distance :math:`s'`.
    :rtype: float or Tensor
    """
    if diopter is None:
        if fl_img is None:
            fl_img = fl_obj
        return objd(obj_d, fl_img, fl_obj)
    else:
        n_obj, n_img = fl_obj, fl_img
        return objd(obj_d, n_img, n_obj, diopter)


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
