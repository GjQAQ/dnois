############################
Unit convention
############################
Physical quantities stored in computer do not have a unit with themselves,
no matter they are represented as an ``int``, ``float``, or ``torch.Tensor``.
The unit of a physical quantity is determined by the context.
To this end, DNOIS provides a unit system based on SI to handle physical quantities.

In DNOIS, a physical unit is represented as a string of its symbol. For example, meter is
represented as ``"m"`` and radian is represented as ``"rad"``. Commonly used prefixes
are supported, such as ``"k"`` for kilo and ``"m"`` for milli. For example, kilometer is
represented as ``"km"`` and millimeter is represented as ``"mm"``. All supported units
are listed in :ref:`ref_dnois_units`.

DNOIS maintains a global **default unit list** for all physical quantities.
For example, the default unit for length is meter and the default unit for angle is radian.
Default units can be accessed via :func:`dnois.get_default` and :func:`dnois.set_default`.
Units of quantities in DNOIS is the corresponding default unit if not specified otherwise.

Note that changing the default unit does not affect quantities already created.
It dictates only how DNOIS interprets given numerics. For example, assuming one creates a
:class:`~dnois.optics.rt.Spherical` with ``roc=1.`` when default unit for length is meter
and then modifies the default unit to millimeter, its ROC will not be ``0.001`` but still ``1.``.
However, its interpretation will change from ``1m`` to ``1mm``.

.. caution::
    Do not expect existing quantities to change their unit when default units are changed.

One of reasons why some global default units have to be maintained is that
it is hard to maintain a unit for all ``float`` and ``torch.Tensor``.
With global default units, all quantities with same dimension are assumed to have the same unit.
For example, all lengths are assumed to be in meter, or millimeter, etc.
One counterexample is wavelengths in :mod:`dnois.mt`, because dispersion equations
are calculated in unified unit. As a result, units of wavelengths c

Lastly, units of same dimension can be converted to each other via :func:`dnois.convert`.
If two units with different dimensions are given, an :exc:`ValueError` will be raised.
