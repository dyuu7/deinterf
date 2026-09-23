from deinterf.compensator.tmi.linear._term import (
    Eddy,
    Eddy8,
    Eddy9,
    Induced,
    Induced5,
    Induced6,
    Permanent,
)


class Terms:
    """Predefined terms for linear TMI compensation.

    ``Terms_16`` combines 3 permanent, 5 induced, and 8 eddy current features.
    ``Terms_18`` combines 3 permanent, 6 induced, and 9 eddy current features.
    ``Induced`` and ``Eddy`` use the 5- and 8-feature variants, respectively.

    Combine terms with ``|`` or select a data source with ``[index]``.
    """

    Permanent = Permanent()

    Induced_5 = Induced5()
    Induced_6 = Induced6()
    Induced = Induced()

    Eddy_8 = Eddy8()
    Eddy_9 = Eddy9()
    Eddy = Eddy()

    Terms_16 = Permanent | Induced_5 | Eddy_8
    Terms_18 = Permanent | Induced_6 | Eddy_9
