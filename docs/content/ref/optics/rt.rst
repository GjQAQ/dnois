#####################################
Ray Tracing
#####################################
.. automodule:: dnois.optics.rt

*********************************
Characterizing rays
*********************************
.. autosummary::
    :toctree: ../../generated/optics/rt/ray

    BatchedRay
    NoValidRayError

**********************************
Optical surfaces
**********************************
.. autosummary::
    :toctree: ../../generated/optics/rt/surf

    paraxialize
    surface_types
    CoaxialContext
    CoaxialSurfaceSequence
    Context
    CircularSurface
    RayCollector
    Surface
    SurfaceSequence

Apertures
============================================
.. autosummary::
    :toctree: ../../generated/optics/rt/surf/aperture

    AnnularAperture
    Aperture
    BoundedAperture
    CircularAperture
    DummyAperture

Specific surface types
=================================
.. autosummary::
    :toctree: ../../generated/optics/rt/surf/types

    AsphericalRadialPhase
    CircularStop
    Conic
    EvenAspherical
    Fresnel
    Grating
    Planar
    PolynomialPhase
    Spherical
    Stop
    ThinLens
    Zernike

************************************
Ray-tracing-based optical systems
************************************
.. autosummary::
    :toctree: ../../generated/optics/rt/sys

    CoaxialRayTracing
    CRTSpotDiagram
    CRTVisConfig

*************************************************************
Configuration for determining ray-surface intersection
*************************************************************
.. autosummary::
    :toctree: ../../generated/optics/rt/config

    IntersectionConfig
