from datetime import datetime, timedelta
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import ppigrf
from dafmit_aeromag import Dataset, Selection
from dataioc import DataIoC, DataNDArray, UniqueData
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.foundation.sensors import DirectionalCosine, MagVector, Tmi
from deinterf.metrics.fom import improve_rate
from deinterf.utils.transform import magvec2dircosine


class LocationWGS84(DataNDArray, UniqueData):
    """Mark location data as unique."""

    def __new__(cls, lon: ArrayLike, lat: ArrayLike, alt: ArrayLike):
        return super().__new__(cls, lon, lat, alt)


class Date(NamedTuple):
    """Non-indexable data is unique by default."""

    year: int  # Calendar year.
    doy: int  # Day of year.


class IGRF(DataNDArray):
    @classmethod
    def __build__(cls, container: DataIoC):
        lon, lat, alt = container[LocationWGS84].T

        # Convert day of year to a datetime.
        year, doy = container[Date]
        date = datetime(year, 1, 1) + timedelta(days=doy - 1)

        geo_e, geo_n, geo_u = ppigrf.igrf(lon, lat, alt / 1000, date)
        geo = np.vstack((geo_e, geo_n, geo_u)).T

        return cls(*geo.T)


class InertialAttitude(DataNDArray):
    def __new__(cls, yaw: ArrayLike, pitch: ArrayLike, roll: ArrayLike):
        return super().__new__(cls, yaw, pitch, roll)


class InsDirectionalCosine(DirectionalCosine):
    @classmethod
    def __build__(cls, container: DataIoC) -> DirectionalCosine:
        # The input order is yaw, pitch, roll in the DEN convention.
        att_angle = container[InertialAttitude]
        # Convert attitude angles from DEN to ENU.
        att_angle = att_angle[:, [1, 2, 0]]
        att_angle[:, 2] = -att_angle[:, 2]

        r = R.from_euler("xyz", att_angle, degrees=True)
        geo_bodyframe = r.apply(container[IGRF], inverse=True)
        dcos = magvec2dircosine(geo_bodyframe)
        return cls(*dcos.T)


if __name__ == "__main__":
    flt_d = Dataset().read(
        Selection(1002, lines="1002.02"),
        columns=[
            "flux_d_x",
            "flux_d_y",
            "flux_d_z",
            "mag_3_uc",
            "ins_yaw",
            "ins_pitch",
            "ins_roll",
            "lon",
            "lat",
            "utm_z",
        ],
        split="train",
    )

    # Extract the flight date.
    year, doy = int(flt_d["year"].iloc[0]), int(flt_d["doy"].iloc[0])

    # Run classical compensation.
    tmi_with_interf = Tmi(tmi=flt_d["mag_3_uc"])
    fom_data = DataIoC().with_data(
        MagVector[1](bx=flt_d["flux_d_x"], by=flt_d["flux_d_y"], bz=flt_d["flux_d_z"]),
    )

    compensator = TollesLawson(terms=Terms.Terms_16[1])
    tmi_clean_classic = compensator.fit_transform(fom_data, tmi_with_interf)

    # Run compensation with INS-derived direction cosines.
    fom_data = DataIoC().with_data(
        Date(year=year, doy=doy),
        LocationWGS84(lon=flt_d["lon"], lat=flt_d["lat"], alt=flt_d["utm_z"]),
        InertialAttitude[1](
            yaw=flt_d["ins_yaw"], pitch=flt_d["ins_pitch"], roll=flt_d["ins_roll"]
        ),
        MagVector[1](bx=flt_d["flux_d_x"], by=flt_d["flux_d_y"], bz=flt_d["flux_d_z"]),
    )
    # Use INS-derived direction cosines as the data source.
    fom_data.add_provider(DirectionalCosine, InsDirectionalCosine)
    tmi_clean_ins = compensator.fit_transform(fom_data, tmi_with_interf)

    # Compare compensation performance.
    ir_classic = improve_rate(tmi_with_interf, tmi_clean_classic, verbose=True)
    print(f"{ir_classic=}")
    ir_ins = improve_rate(tmi_with_interf, tmi_clean_ins, verbose=True)
    print(f"{ir_ins=}")

    # Plot the input and compensated signals.
    plt.plot(tmi_with_interf, label="tmi_with_interf")
    plt.plot(tmi_clean_classic, label="tmi_clean_classic")
    plt.plot(tmi_clean_ins, label="tmi_clean_ins")
    plt.legend()
    plt.show()
