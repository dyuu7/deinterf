from __future__ import annotations

import numpy as np
from dataioc import DataIoC

from deinterf.foundation import ComposableTerm
from deinterf.foundation.sensors import DirectionalCosine, MagIntensity


class Permanent(ComposableTerm):
    def __build__(self, container: DataIoC) -> DirectionalCosine:
        return container[DirectionalCosine]


class Induced6(ComposableTerm):
    def __build__(self, container: DataIoC) -> np.ndarray:
        intensity = container[MagIntensity]
        cos_x, cos_y, cos_z = container[DirectionalCosine].T
        # (n,) * (n, 6) -> (n, 6)
        feats = intensity[:, None] * np.column_stack(
            (
                cos_x * cos_x,
                cos_x * cos_y,
                cos_x * cos_z,
                cos_y * cos_y,  # removed in Induced_5 version
                cos_y * cos_z,
                cos_z * cos_z,
            )
        )
        return feats


class Induced5(ComposableTerm):
    def __build__(self, container: DataIoC) -> np.ndarray:
        feats = container[Induced6]
        feats = np.delete(feats, feats.shape[1] // 2, 1)
        return feats


class Induced(Induced5):
    ...


class Eddy9(ComposableTerm):
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
                cos_y * cos_y_dot,  # removed in Eddy_8 version
                cos_y * cos_z_dot,
                cos_z * cos_x_dot,
                cos_z * cos_y_dot,
                cos_z * cos_z_dot,
            )
        )
        return feats


class Eddy8(ComposableTerm):
    def __build__(self, container: DataIoC) -> np.ndarray:
        feats = container[Eddy9]
        feats = np.delete(feats, feats.shape[1] // 2, 1)
        return feats


class Eddy(Eddy8): 
    ...
