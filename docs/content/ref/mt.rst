###########################
Materials
###########################

.. automodule:: dnois.mt

***************************
Base class
***************************
.. autoclass:: Material
    :members: n

****************************
Materials
****************************
.. autosummary::
    :toctree: ../generated/mt

    Cauchy
    Conrady
    Constant
    Herzberger
    Schott
    Sellmeier1
    Sellmeier2
    Sellmeier3
    Sellmeier4
    Sellmeier5

*******************************
Built-in materials
*******************************

Built-in materials can be accessed without instantiation or calling :py:func:`get`.
Data of materials come from `<https://refractiveindex.info/>`_ if not mentioned otherwise.

.. autoattribute:: dnois.mt.vacuum

.. autoattribute:: dnois.mt.silica

.. _accessing_materials:

*****************************
Accessing materials
*****************************
.. autosummary::
    :toctree: ../generated/mt

    dispersion_types
    get
    is_available
    list_all
    register
    remove
    update

*****************************
Others
*****************************
.. autofunction:: dnois.mt.refractive_index
