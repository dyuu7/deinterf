<!-- markdownlint-disable -->

<div align="center">

<img alt="LOGO" src="./docs/assets/logo.svg" width="80%" />

<!-- markdownlint-restore -->
</div>

## Intro

Based on the Tolles-Lawson (T-L) aeromagnetic compensation model, this tool compensates for the aircraft's own magnetic interference in aeromagnetic survey data. Future enhancements may include magnetic vector compensation, nonlinear methods, neural network approaches, etc. This tool is designed using the inversion of control concept and supports the extension of additional magnetic interference components.

## Getting started

`deinterf` can be installed using pip, tested on python 3.9:

```shell
pip install deinterf
```

## Use Cases

Classical T-L compensation:

```python

import matplotlib.pyplot as plt
from sgl2020 import Sgl2020

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.foundation.sensors import MagVector, Tmi
from deinterf.metrics.fom import improve_rate, noise_level
from deinterf.utils.data_ioc import DataIoC

if __name__ == "__main__":
    surv_d = (
        Sgl2020()
        .line(["1002.02"])
        .source(
            [
                "flux_b_x",
                "flux_b_y",
                "flux_b_z",
                "mag_3_uc",
            ]
        )
        .take()
    )
    flt_d = surv_d["1002.02"]

    # prepare data
    tmi_with_interf = Tmi(tmi=flt_d["mag_3_uc"])
    fom_data = DataIoC().add(
        MagVector(bx=flt_d["flux_b_x"], by=flt_d["flux_b_y"], bz=flt_d["flux_b_z"])
    )

    # create compensator
    compensator = TollesLawson(terms=Terms.Terms_16)
    compensator.fit(fom_data, tmi_with_interf)

    # compensate new data
    # input the `DataIoC` corresponding to the new flight path and the signal to be compensated.
    tmi_clean = compensator.transform(fom_data, tmi_with_interf)

    # if it is FOM data, it can be fitted and compensated in one step
    tmi_clean = compensator.fit_transform(fom_data, tmi_with_interf)

    # predict magnetic interference only
    interf = compensator.predict(fom_data)

    # evaluating compensation performance
    comped_noise_level = noise_level(tmi_clean)
    print(f"{comped_noise_level=}")

    ir = improve_rate(tmi_with_interf, tmi_clean)
    print(f"{ir=}")

    # simple plot signals
    plt.plot(tmi_with_interf, label="tmi_with_interf")
    plt.plot(tmi_clean, label="tmi_clean")
    plt.legend()
    plt.show()
```

Using "direction cosines calculated by the inertial navigation system (INS) instead of the magnetic vector" as an example, this section demonstrates how to extend or modify the classic T-L model:

```python
from datetime import datetime, timedelta
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import ppigrf
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R
from sgl2020 import Sgl2020

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.foundation.sensors import DirectionalCosine, MagVector, Tmi
from deinterf.metrics.fom import improve_rate
from deinterf.utils.data_ioc import DataIoC, DataNDArray, UniqueData
from deinterf.utils.transform import magvec2dircosine


class LocationWGS84(DataNDArray, UniqueData):
    """Explicitly specified as unique data, which means that dependencies with different numbers share the same data source at build time
    """
    def __new__(cls, lon: ArrayLike, lat: ArrayLike, alt: ArrayLike):
        return super().__new__(cls, lon, lat, alt)


class Date(NamedTuple):
    """Non-indexable type, default is `UniqueData`
    """
    year: int  # year
    doy: int  # day of year


class IGRF(DataNDArray):
    @classmethod
    def __build__(cls, container: DataIoC):
        lon, lat, alt = container[LocationWGS84].T

        # doy to datetime
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
        att_angle = container[InertialAttitude]  # (yaw, pitch, roll): DEN
        # DEN to ENU
        att_angle = att_angle[:, [1, 2, 0]]
        att_angle[:, 2] = -att_angle[:, 2]

        r = R.from_euler("xyz", att_angle, degrees=True)
        geo_bodyframe = r.apply(container[IGRF], inverse=True)
        dcos = magvec2dircosine(geo_bodyframe)
        return cls(*dcos.T)


if __name__ == "__main__":
    surv_d = (
        Sgl2020()
        .line(["1002.02"])
        .source(
            [
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
            ]
        )
        .take()
    )
    flt_d = surv_d["1002.02"]

    # date of flt1002
    year, doy = 2020, 172

    # classic compensation
    tmi_with_interf = Tmi(tmi=flt_d["mag_3_uc"])
    fom_data = DataIoC().with_data(
        MagVector[1](bx=flt_d["flux_d_x"], by=flt_d["flux_d_y"], bz=flt_d["flux_d_z"]),
    )

    compensator = TollesLawson(terms=Terms.Terms_16[1])
    tmi_clean_classic = compensator.fit_transform(fom_data, tmi_with_interf)

    # INS extended compensation
    fom_data = DataIoC().with_data(
        Date(year=year, doy=doy),
        LocationWGS84(lon=flt_d["lon"], lat=flt_d["lat"], alt=flt_d["utm_z"]),
        InertialAttitude[1](yaw=flt_d["ins_yaw"], pitch=flt_d["ins_pitch"], roll=flt_d["ins_roll"]),
        MagVector[1](bx=flt_d["flux_d_x"], by=flt_d["flux_d_y"], bz=flt_d["flux_d_z"]),
    )
    # modify the data source that Direction Cosines depends on
    fom_data.add_provider(DirectionalCosine, InsDirectionalCosine)
    tmi_clean_ins = compensator.fit_transform(fom_data, tmi_with_interf)

    # comparing the two models
    ir_classic = improve_rate(tmi_with_interf, tmi_clean_classic, verbose=True)
    print(f"{ir_classic=}")
    ir_ins = improve_rate(tmi_with_interf, tmi_clean_ins, verbose=True)
    print(f"{ir_ins=}")

    plt.plot(tmi_with_interf, label="tmi_with_interf")
    plt.plot(tmi_clean_classic, label="tmi_clean_classic")
    plt.plot(tmi_clean_ins, label="tmi_clean_ins")
    plt.legend()
    plt.show()
```

## Acknowledgements

This project exists thanks to all the people who contribute. Thank you to our wonderful contributors:

<a href="https://github.com/dorian-li/deinterf/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=dyuu7/deinterf" />
</a>

## Licensing

The code in this project is licensed under MIT license.
