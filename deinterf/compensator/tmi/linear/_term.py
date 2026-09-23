from __future__ import annotations

import numpy as np
from dataioc import DataIoC

from deinterf.foundation import ComposableTerm
from deinterf.foundation.sensors import DirectionalCosine, MagIntensity


class Permanent(ComposableTerm):
    """Three permanent-field features given by the direction cosines."""

    def __build__(self, container: DataIoC) -> DirectionalCosine:
        return container[DirectionalCosine]


class Induced6(ComposableTerm):
    """Six quadratic direction-cosine features scaled by field intensity."""

    def __build__(self, container: DataIoC) -> np.ndarray:
        intensity = container[MagIntensity]
        cos_x, cos_y, cos_z = container[DirectionalCosine].T
        # Scale all six feature columns by the intensity of each sample.
        feats = intensity[:, None] * np.column_stack(
            (
                cos_x * cos_x,
                cos_x * cos_y,
                cos_x * cos_z,
                cos_y * cos_y,  # Omitted in Induced5.
                cos_y * cos_z,
                cos_z * cos_z,
            )
        )
        return feats


class Induced5(ComposableTerm):
    """Five induced-field features, omitting the squared y direction cosine."""

    def __build__(self, container: DataIoC) -> np.ndarray:
        feats = container[Induced6]
        feats = np.delete(feats, feats.shape[1] // 2, 1)
        return feats


class Induced(Induced5):
    """Default induced-field term using the five-feature variant."""

    ...


class Eddy9(ComposableTerm):
    """Nine eddy current features from direction cosines and their derivatives.

    Derivatives use unit sample spacing, and all features are scaled by field
    intensity.
    """

    def __build__(self, container: DataIoC) -> np.ndarray:
        intensity = container[MagIntensity]
        cos_x, cos_y, cos_z = container[DirectionalCosine].T
        cos_x_dot = np.gradient(cos_x)
        cos_y_dot = np.gradient(cos_y)
        cos_z_dot = np.gradient(cos_z)
        feats = intensity[:, None] * np.column_stack(
            (
                cos_x * cos_x_dot,
                cos_x * cos_y_dot,
                cos_x * cos_z_dot,
                cos_y * cos_x_dot,
                cos_y * cos_y_dot,  # Omitted in Eddy8.
                cos_y * cos_z_dot,
                cos_z * cos_x_dot,
                cos_z * cos_y_dot,
                cos_z * cos_z_dot,
            )
        )
        return feats


class Eddy8(ComposableTerm):
    """Eight eddy current features, omitting the y cosine times its derivative."""

    def __build__(self, container: DataIoC) -> np.ndarray:
        feats = container[Eddy9]
        feats = np.delete(feats, feats.shape[1] // 2, 1)
        return feats


class Eddy(Eddy8): 
    """Default eddy current term using the eight-feature variant."""

    ...
