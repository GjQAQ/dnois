#########################################
Sequential Coaxial System
#########################################

.. currentmodule:: dnois.optics.rt

****************************************
Optical system model
****************************************
.. autosummary::
    :toctree: ../../generated/optics/rt/crt

    CoaxialRayTracing

*******************************************
Point spread function models
*******************************************
There are various different ways to compute PSF of a :class:`CoaxialRayTracing` model,
each of which is represented as a subclass of :class:`CrtPsfModel`.
They implements the :meth:`CrtPsfModel.psf` method to compute PSF of a given point.

.. autosummary::
    :toctree: ../../generated/optics/rt/crt

    CrtPsfModel
    CoherentKirchoffPsf
    CoherentHuygensPsf
    CoherentFraunhoferPsf
    IncoherentRectKernelPsf
    IncoherentGaussianKernelPsf

Determining PSF center
==============================================
.. autosummary::
    :toctree: ../../generated/optics/rt/crt

    PsfCenterDeterm
    LinearPsfCenter
    FixedPsfCenter
    ChiefRayPsfCenter
    MeanPsfCenter
    RobustMeanPsfCenter

********************************************
Visualization
********************************************
.. autosummary::
    :toctree: ../../generated/optics/rt/crt

    CRTVisConfig
    CRTSpotDiagram