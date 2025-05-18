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

    Air
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
    registered
    remove
    update

*****************************
Others
*****************************
.. autofunction:: dnois.mt.refractive_index
