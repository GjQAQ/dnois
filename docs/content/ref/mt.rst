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
    ConstantRelative
    Herzberger
    Schott
    Sellmeier1
    Sellmeier2
    Sellmeier3
    Sellmeier4
    Sellmeier5

.. _accessing_materials:

*****************************
Accessing materials
*****************************
.. autosummary::
    :toctree: ../generated/mt

    dispersion_types
    get
    lib
    list_all
    register
    registered
    remove
    update

    MaterialNotFoundError

*****************************
Others
*****************************
.. autofunction:: dnois.mt.refractive_index
