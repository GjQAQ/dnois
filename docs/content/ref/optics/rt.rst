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
    RayCollector
    RelativeContext
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
    RectangularAperture

Specific surface types
=================================
.. autosummary::
    :toctree: ../../generated/optics/rt/surf/types

    AsphereRadialPhase
    CircularStop
    Conic
    EvenAsphere
    Fresnel
    Grating
    Plane
    PolynomialPhase
    Sphere
    Stop
    ThinLens
    Zernike

************************************
Ray-tracing-based optical systems
************************************
:class:`CoaxialRayTracingSystem` provides some methods to simulate, optimize, analyze
and visualize **coaxial optical systems** modeled in a **sequential** manner.
It is an optical system model used in most cases.
See the following page for documentation related to it.

.. toctree::
    :maxdepth: 1

    crt

*************************************************************
Configuration for determining ray-surface intersection
*************************************************************
.. autosummary::
    :toctree: ../../generated/optics/rt/config

    IntersectionConfig
