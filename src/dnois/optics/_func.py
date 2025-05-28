import math

import torch

from .. import base, utils
from ..base import typing
from ..base.typing import Numeric, Ts, overload

__all__ = [
    'circle_of_confusion',
    'fresnel_st',
    'fresnel_sr',
    'fresnel_pt',
    'fresnel_pr',
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

    .. warning::
        The behavior of this function when any of the arguments is infinite is undefined.

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

        .. note::
            ``fl_obj`` will be returned if ``img_d`` is infinite.

        .. warning::
            The behavior of this function when ``fl_obj`` or ``fl_img`` is infinite is undefined.

        :param img_d: Image distance :math:`s'`.
        :type img_d: float or Tensor
        :param fl_obj: Object focal length :math:`f`.
        :type fl_obj: float or Tensor
        :param fl_img: Image focal length :math:`f'`. Default: identical to ``fl_obj``.
        :type fl_img: float or Tensor

    -------------------

    .. function:: objd(img_d, n_obj, n_img, diopter)
        :no-index:

        .. math::
            s=\frac{n_1s'}{\phi s'-n_2}

        .. note::
            ``n_obj / diopter`` will be returned if ``img_d`` is infinite.

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
    if diopter is None:  # given two focal lengths
        if fl_img is None:
            fl_img = fl_obj
        return utils.InfinityCond(  # 1.if img_d is infinite,
            lambda x: utils.Conditional(
                lambda y: y == 0,  # 1.else 2.if x - fl_img is zero,
                lambda _: float('inf'),  # 2.then return inf
                lambda y: fl_obj * x / y,  # 2.else return fl_obj * x / (x - fl_img)
                expr_true_tensor=lambda y: torch.full_like(y, float('inf')),
            )(x - fl_img),
            lambda x: fl_obj,  # 1.then return fl_obj
        )(img_d)
    else:  # given two refractive indices and diopter
        n_obj, n_img = fl_obj, fl_img
        return utils.InfinityCond(  # similar logic
            lambda x: utils.Conditional(
                lambda y: y == 0,
                lambda _: float('inf'),
                lambda y: n_obj * x / y,
                expr_true_tensor=lambda y: torch.full_like(y, float('inf')),
            )(diopter * x - n_img),
            lambda x: n_obj / diopter,
        )(img_d)


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

        .. note::
            ``fl_img`` will be returned if ``obj_d`` is infinite.

        .. warning::
            The behavior of this function when ``fl_obj`` or ``fl_img`` is infinite is undefined.

        :param obj_d: Object distance :math:`s`.
        :type obj_d: float or Tensor
        :param fl_obj: Object focal length :math:`f`.
        :type fl_obj: float or Tensor
        :param fl_img: Image focal length :math:`f'`. Default: identical to ``fl_obj``.
        :type fl_img: float or Tensor

    -------------------------

    .. function:: imgd(obj_d, n_obj, n_img, diopter)
        :no-index:

        .. math::
            s'=\frac{n_2s}{\phi s-n_1}

        .. note::
            ``n_img / diopter`` will be returned if ``obj_d`` is infinite.


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


def snell(incident_angle: Numeric, n1: Numeric, n2: Numeric) -> Numeric:
    r"""
    Computes angle of refracted ray :math:`\theta_t` given that of incident ray :math:`\theta_i`
    and refractive indices of both media :math:`n_1` and :math:`n_2`:

    .. math::
        \sin\theta_t=\frac{n_1}{n_2}\sin\theta_i

    .. seealso::
        Use :func:`dnois.refract` if direction of incident and refracted rays
        and normal vector are all vectors.

    :param incident_angle: Angle of incident ray :math:`\theta_i`.
    :param n1: Refractive index :math:`n_1` in incident medium.
    :param n2: Refractive index :math:`n_2` in refractive medium.
    :return: Angle of refracted ray :math:`\theta_t`.
    """
    incident_angle = base.Angle.default_to(incident_angle, 'rad')
    sin_incident = utils.GenericCompute(math.sin, torch.sin)(incident_angle)
    sin_refracted = sin_incident * n1 / n2
    refracted_angle = utils.GenericCompute(math.asin, torch.asin)(sin_refracted)
    return base.Angle.as_default(refracted_angle, 'rad')


def _fresnel_dispatch(func, args, kwargs):
    ba = utils.get_bound_args(func, False, *args, **kwargs)
    overloads = typing.get_overloads(func)
    if len(ba.arguments) == 2:  # form 1
        return typing.cast(Numeric, overloads[0](*args, **kwargs))
    elif 'n1' in ba.kwargs:  # form 2
        return typing.cast(Numeric, overloads[1](*args, **kwargs))
    else:  # form 3
        return typing.cast(Numeric, overloads[2](*args, **kwargs))


def _fresnel_st(i, t):
    sin_t = utils.GenericCompute(math.sin, torch.sin)(t)
    cos_i = utils.GenericCompute(math.cos, torch.cos)(i)
    sin = utils.GenericCompute(math.sin, torch.sin)(i + t)
    return 2 * sin_t * cos_i / sin


def _fresnel_sr(i, t):
    numerator = utils.GenericCompute(math.sin, torch.sin)(t - i)
    denominator = utils.GenericCompute(math.sin, torch.sin)(i + t)
    return numerator / denominator


def _fresnel_pt(i, t):
    sin_t = utils.GenericCompute(math.sin, torch.sin)(t)
    cos_i = utils.GenericCompute(math.cos, torch.cos)(i)
    sin = utils.GenericCompute(math.sin, torch.sin)(i + t)
    cos = utils.GenericCompute(math.cos, torch.cos)(i - t)
    return 2 * sin_t * cos_i / (sin * cos)


def _fresnel_pr(i, t):
    numerator = utils.GenericCompute(math.tan, torch.tan)(i - t)
    denominator = utils.GenericCompute(math.tan, torch.tan)(i + t)
    return numerator / denominator


def _fresnel_form1(impl, incident_angle: Numeric, refracted_angle: Numeric) -> Numeric:
    i, t = incident_angle, refracted_angle
    i = base.Angle.default_to(i, 'rad')
    t = base.Angle.default_to(t, 'rad')
    return impl(i, t)


def _fresnel_form2(normal_expr, impl, incident_angle: Numeric, n1: Numeric, n2: Numeric) -> Numeric:
    def _non_normal(x):
        t = snell(x, n1, n2)
        return _fresnel_form1(impl, x, t)

    c = utils.Conditional(lambda x: x == 0., normal_expr(n1, n2), _non_normal)(incident_angle)
    return c


@overload  # noqa
def fresnel_st(incident_angle: Numeric, refracted_angle: Numeric) -> Numeric:
    return _fresnel_form1(_fresnel_st, incident_angle, refracted_angle)


@overload
def fresnel_st(incident_angle: Numeric, *, n1: Numeric, n2: Numeric) -> Numeric:
    return _fresnel_form2(lambda a, b: 2 * a / (a + b), _fresnel_st, incident_angle, n1, n2)


def fresnel_st(*args, **kwargs) -> Numeric:
    r"""
    Computes Fresnel's equation for s-polarized transmitted (refractive) wave:

    .. math::
        \frac{E_\text{s,t}}{E_\text{s,i}}=\frac{2\sin\theta_\text{t}\cos\theta_\text{i}}
        {\sin(\theta_\text{i}+\theta_\text{t})}

    .. function:: fresnel_st(incident_angle, refracted_angle)
        :no-index:

        .. caution::
            This version cannot handle normal incidence, i.e. :math:`\theta_i=\theta_t=0`
            and hence the denominator is zero.

        :param incident_angle: Incident angle :math:`\theta_\text{i}`.
        :param refracted_angle: Refracted angle :math:`\theta_\text{t}`.

    .. function:: fresnel_st(incident_angle, *, n1, n2)
        :no-index:

        .. note::
            This version handles normal incidence, i.e. :math:`\theta_i=\theta_t=0` correctly,
            in which case the result is :math:`2n_1/(n_1+n_2)`.

        :param incident_angle: Incident angle :math:`\theta_\text{i}`.
        :param n1: Refractive index :math:`n_1` in incident medium.
        :param n2: Refractive index :math:`n_2` in refractive medium.

    :return: Ratio of electric intensity amplitude of transmitted wave to that of incident wave.
    """
    return _fresnel_dispatch(fresnel_st, args, kwargs)


@overload
def fresnel_sr(incident_angle: Numeric, refracted_angle: Numeric) -> Numeric:
    return _fresnel_form1(_fresnel_sr, incident_angle, refracted_angle)


@overload
def fresnel_sr(incident_angle: Numeric, *, n1: Numeric, n2: Numeric) -> Numeric:
    return _fresnel_form2(lambda a, b: (b - a) / (a + b), _fresnel_sr, incident_angle, n1, n2)


def fresnel_sr(*args, **kwargs) -> Numeric:
    r"""
    Computes Fresnel's equation for s-polarized reflected wave:

    .. math::
        \frac{E_\text{s,r}}{E_\text{s,i}}=-\frac{\sin(\theta_\text{i}-\theta_\text{t})}
        {\sin(\theta_\text{i}+\theta_\text{t})}

    .. function:: fresnel_sr(incident_angle, refracted_angle)
        :no-index:

        .. caution::
            This version cannot handle normal incidence, i.e. :math:`\theta_i=\theta_t=0`
            and hence the denominator is zero.

        :param incident_angle: Incident angle :math:`\theta_\text{i}`.
        :param refracted_angle: Refracted angle :math:`\theta_\text{t}`.

    .. function:: fresnel_sr(incident_angle, *, n1, n2)
        :no-index:

        .. note::
            This version handles normal incidence, i.e. :math:`\theta_i=\theta_t=0` correctly,
            in which case the result is :math:`(n_2-n_1)/(n_2+n_1)`.

        :param incident_angle: Incident angle :math:`\theta_\text{i}`.
        :param n1: Refractive index :math:`n_1` in incident medium.
        :param n2: Refractive index :math:`n_2` in refractive medium.

    :return: Ratio of electric intensity amplitude of reflected wave to that of incident wave.
    """
    return _fresnel_dispatch(fresnel_sr, args, kwargs)


@overload  # noqa
def fresnel_pt(incident_angle: Numeric, refracted_angle: Numeric) -> Numeric:
    return _fresnel_form1(_fresnel_pt, incident_angle, refracted_angle)


@overload
def fresnel_pt(incident_angle: Numeric, *, n1: Numeric, n2: Numeric) -> Numeric:
    return _fresnel_form2(lambda a, b: 2 * a / (a + b), _fresnel_pt, incident_angle, n1, n2)


def fresnel_pt(*args, **kwargs) -> Numeric:
    r"""
    Computes Fresnel's equation for p-polarized transmitted (refractive) wave:

    .. math::
        \frac{E_\text{p,t}}{E_\text{p,i}}=\frac{2\sin\theta_\text{t}\cos\theta_\text{i}}
        {\sin(\theta_\text{i}+\theta_\text{t})\cos(\theta_\text{i}-\theta_\text{t})}

    .. function:: fresnel_pt(incident_angle, refracted_angle)
        :no-index:

        .. caution::
            This version cannot handle normal incidence, i.e. :math:`\theta_i=\theta_t=0`
            and hence the denominator is zero.

        :param incident_angle: Incident angle :math:`\theta_\text{i}`.
        :param refracted_angle: Refracted angle :math:`\theta_\text{t}`.

    .. function:: fresnel_pt(incident_angle, *, n1, n2)
        :no-index:

        .. note::
            This version handles normal incidence, i.e. :math:`\theta_i=\theta_t=0` correctly,
            in which case the result is :math:`2n_1/(n_1+n_2)`.

        :param incident_angle: Incident angle :math:`\theta_\text{i}`.
        :param n1: Refractive index :math:`n_1` in incident medium.
        :param n2: Refractive index :math:`n_2` in refractive medium.

    :return: Ratio of electric intensity amplitude of transmitted wave to that of incident wave.
    """
    return _fresnel_dispatch(fresnel_pt, args, kwargs)


@overload
def fresnel_pr(incident_angle: Numeric, refracted_angle: Numeric) -> Numeric:
    return _fresnel_form1(_fresnel_pr, incident_angle, refracted_angle)


@overload
def fresnel_pr(incident_angle: Numeric, *, n1: Numeric, n2: Numeric) -> Numeric:
    return _fresnel_form2(lambda a, b: (b - a) / (a + b), _fresnel_pr, incident_angle, n1, n2)


def fresnel_pr(*args, **kwargs) -> Numeric:
    r"""
    Computes Fresnel's equation for p-polarized reflected wave:

    .. math::
        \frac{E_\text{p,r}}{E_\text{p,i}}=\frac{\tan(\theta_\text{i}-\theta_\text{t})}
        {\tan(\theta_\text{i}+\theta_\text{t})}

    .. function:: fresnel_pr(incident_angle, refracted_angle)
        :no-index:

        .. caution::
            This version cannot handle normal incidence, i.e. :math:`\theta_i=\theta_t=0`
            and hence the denominator is zero.

        :param incident_angle: Incident angle :math:`\theta_\text{i}`.
        :param refracted_angle: Refracted angle :math:`\theta_\text{t}`.

    .. function:: fresnel_pr(incident_angle, *, n1, n2)
        :no-index:

        .. note::
            This version handles normal incidence, i.e. :math:`\theta_i=\theta_t=0` correctly,
            in which case the result is :math:`(n_2-n_1)/(n_2+n_1)`.

        :param incident_angle: Incident angle :math:`\theta_\text{i}`.
        :param n1: Refractive index :math:`n_1` in incident medium.
        :param n2: Refractive index :math:`n_2` in refractive medium.

    :return: Ratio of electric intensity amplitude of reflected wave to that of incident wave.
    """
    return _fresnel_dispatch(fresnel_pr, args, kwargs)
