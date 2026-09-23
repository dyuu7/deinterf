from __future__ import annotations

import numpy as np
from dataioc import DataIoC, DataNDArray
from numpy.typing import ArrayLike

from deinterf.utils.transform import magvec2dircosine, magvec2intensity


class MagVector(DataNDArray):
    """Magnetic field vectors of shape (n_samples, 3), in x, y, z order."""

    def __new__(cls, bx: ArrayLike, by: ArrayLike, bz: ArrayLike, **kwargs):
        """Stack the x, y, and z component arrays as columns."""
        return super().__new__(cls, bx, by, bz, **kwargs)

    @property
    def bx(self):
        """Magnetic field component along the x axis."""
        return self[:, 0]

    @property
    def by(self):
        """Magnetic field component along the y axis."""
        return self[:, 1]

    @property
    def bz(self):
        """Magnetic field component along the z axis."""
        return self[:, 2]


class MagIntensity(DataNDArray):
    """Magnetic field intensities of shape (n_samples,).

    Values are vector magnitudes derived from ``MagVector`` by default and
    retain the units of its components.
    """

    @classmethod
    def __build__(cls, container: DataIoC):
        intensity = magvec2intensity(container[MagVector])
        return cls(intensity)


class Tmi(DataNDArray):
    """Total magnetic intensity (TMI) samples of shape (n_samples,)."""

    def __new__(cls, tmi: ArrayLike):
        """Wrap TMI samples as an indexed array."""
        return super().__new__(cls, tmi)


class DirectionalCosine(DataNDArray):
    """Direction cosines of shape (n_samples, 3), in x, y, z order.

    Values are unitless and are derived from ``MagVector`` by default.
    """

    def __new__(cls, dir_cosine_x: ArrayLike, dir_cosine_y: ArrayLike, dir_cosine_z: ArrayLike):
        """Stack the x, y, and z direction-cosine arrays as columns."""
        return super().__new__(cls, dir_cosine_x, dir_cosine_y, dir_cosine_z)

    @classmethod
    def __build__(cls, container: DataIoC) -> DirectionalCosine:
        dir_cosine = magvec2dircosine(container[MagVector])
        dir_cosine_x, dir_cosine_y, dir_cosine_z = np.transpose(dir_cosine)
        return cls(
            dir_cosine_x=dir_cosine_x,
            dir_cosine_y=dir_cosine_y,
            dir_cosine_z=dir_cosine_z,
        )

    @property
    def dcosx(self):
        """Direction cosine along the x axis."""
        return self[:, 0]

    @property
    def dcosy(self):
        """Direction cosine along the y axis."""
        return self[:, 1]

    @property
    def dcosz(self):
        """Direction cosine along the z axis."""
        return self[:, 2]
