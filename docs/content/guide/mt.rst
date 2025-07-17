##########################################
Materials
##########################################

Refractive index of materials plays a crucial role in optical design.
Typically, it varies with wavelength (a.k.a. dispersion) and the temperature of the material.
To support the computation of refractive index under various conditions,
DNOIS provides a comprehensive set of dispersion models.

*****************************************
Material objects
*****************************************

In DNOIS, the materials with a specific dispersion formula are represented
by a class that inherits from :class:`~dnois.mt.Material`. For example,
the :class:`~dnois.mt.Cauchy` class represents materials described by Cauchy's dispersion formula.
An instance of these classes corresponds to a specific material type.
For example, you can retrieve the material object of N-BK7 by ``dnois.mt.get('N-BK7')``,
which is an instance of :class:`~dnois.mt.Sellmeier1`.

Note that refractive indices are dependent on temperature, hence the dispersion formula
only holds under a certain temperature, which is called its **reference temperature**.
The argument of the dispersion formula, i.e. wavelength is also measured not in vacuum,
but in air under the reference temperature and 1 atmospheric pressure, which is called
the **reference condition**.

****************************************
Accessing materials
****************************************
In DNOIS you can access materials by their name. For example, to get the material
object of BK7, one can use ``dnois.mt.get('N-BK7')``. Names are case-sensitive.
The reason this is possible is that DNOIS maintains a global material library.
:func:`dnois.mt.register` and :func:`dnois.mt.remove` are used to add and remove
materials from the library. See :ref:`accessing_materials` for more details.

Sometimes materials are imported in groups and two materials from two groups
may have the same name. To distinguish them, one can use a qualifier
seperated by a colon, for example, ``dnois.mt.get('SCHOTT:N-BK7')``
to get the material object of B-BK7 in group SCHOTT.
Materials are allowed to have no qualifier. An empty string
as qualifier is treated as no qualifier.
