###################################
Optical System
###################################

.. automodule:: dnois.optics

.. _modeling_optical_systems:

************************************
Modeling optical systems
************************************

.. _abstract_models:

Abstract models
====================================
TODO

.. autosummary::
    :toctree: ../generated/optics/abstract

    IdealOptics
    PinholeOptics
    PsfImagingOptics

Implemented models
=====================================

The optical response of a given system varies when different ways
are chosen to model it. Currently, there are two categories of methods
to model optical systems: :doc:`optics/df` and :doc:`optics/rt`.
The following content implements :ref:`abstract_models` in these two ways.

.. toctree::
    :maxdepth: 2

    optics/df
    optics/rt

Paraxial optics
=====================================

Behavior of an optical system under paraxial approximation is modeled by :class:`ParaxialSystem`.
It is an abstract class but typically you need only to instantiate it with
:func:`ParaxialSystem.from_interface` rather than explicitly instantiating its subclass.

.. autosummary::
    :toctree: ../generated/optics/paraxial

    ParaxialSystem
    FiniteParaxialSystem
    InfiniteParaxialSystem

.. _image_formation:

***********************************
Image formation
***********************************

.. autosummary::
    :toctree: ../generated/optics/formation

    simple
    depth_aware
    space_variant
    superpose

**************************************
Optics
**************************************
Here are some useful functions for optical calculation.

.. autosummary::
    :toctree: ../generated/optics/o

    circle_of_confusion
    imgd
    objd
    norm_psf

**************************************
Regularization
**************************************
