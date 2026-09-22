<!-- markdownlint-disable -->

<div align="center">

<img alt="LOGO" src="./docs/assets/logo.svg" width="80%" />

<!-- markdownlint-restore -->
</div>

## Intro

Based on the Tolles-Lawson (T-L) aeromagnetic compensation model, this tool compensates for the aircraft's own magnetic interference in aeromagnetic survey data. Future enhancements may include magnetic vector compensation, nonlinear methods, neural network approaches, etc. This tool is designed using the inversion of control concept and supports the extension of additional magnetic interference components.

## Getting started

`deinterf` requires Python 3.10 or newer:

```shell
pip install deinterf
```

The core package accepts your own NumPy-compatible sensor data and uses
[`dataioc`](https://github.com/dyuu7/dataioc) for data dependencies. To run the
flight-data examples below, install their data reader, plotting and IGRF dependencies:

```shell
pip install "deinterf[examples]"
```

For development from this checkout (including all example tests):

```shell
uv sync --extra examples
uv run --extra examples pytest
```

With only `uv sync` and `uv run pytest`, the core tests run and the optional
example tests are skipped. Tests use synthetic local flight files and do not
download data.

## Python compatibility

The supported range is CPython 3.10–3.14 (standard builds with the GIL).
CI checks both the core installation and the `examples` extra:

| Platform | Python versions |
| --- | --- |
| Linux x86_64 | 3.10, 3.11, 3.12, 3.13, 3.14 |
| Windows x86_64 | 3.10, 3.14 |
| macOS ARM64 | 3.10, 3.14 |

Release checks use `uv.lock`, require wheels for third-party runtime dependencies,
and test built wheels and source distributions outside the checkout on Linux
with Python 3.10 and 3.14. Separate checks resolve the lowest direct dependencies
on 3.10 and the latest compatible dependencies on 3.10 and 3.14.

A weekly workflow repeats dependency checks and probes the latest stable CPython.
A successful probe alone does not extend the supported range: new Python versions
must be added to the release matrix. Dependency updates should change the lockfile
deliberately and pass the same release checks.

## Migration

- Import container types directly from `dataioc`; `deinterf.utils.data_ioc`
  and its private modules have been removed.
- Replace `sgl2020.Sgl2020` queries with `dafmit_aeromag.Dataset` and `Selection`,
  specifying the flight and requested columns explicitly.
- Install `deinterf[examples]` for flight-data examples. `dafmit-aeromag`,
  `matplotlib` and `ppigrf` are optional; the former `plot` extra is replaced by
  `examples`.
- Python 3.10 or newer is required.

## Versioning

Git tags (`vX.Y.Z`) are the version source. `hatch-vcs` derives release and
development versions when building the package. `deinterf.__version__` reads
the installed package metadata through `importlib.metadata`; no version file
is generated or maintained in the source tree.

Install the project before importing it. After changing commits or tags, refresh
the version metadata of an editable installation with:

```shell
uv sync --extra examples --reinstall-package deinterf
```

CI fetches the full Git history and tags before installing or building.

## Use Cases

Flight data is loaded through
[`dafmit-aeromag`](https://github.com/dyuu7/dafmit-aeromag), which replaces
`sgl2020`. `Dataset.read(Selection(...), columns=...)` returns a DataFrame;
the examples explicitly select `split="train"` for calibration line `1002.02`
in flight `1002`. Source field names and units are preserved, and `year`/`doy`
come from the reader's identity columns. The examples use 10 Hz sampling.
On first use, the reader downloads and verifies the flight file in its own cache.
Use `Dataset(data_dir=..., offline=True)` to read an existing verified cache;
see the reader's documentation for data access and dataset terms.

The data container is now the independent
[`dataioc`](https://github.com/dyuu7/dataioc) package, primarily implemented by
[yanang007](https://github.com/yanang007), with the original idea proposed by
[dyuu7](https://github.com/dyuu7). `deinterf` uses its NumPy extra and public API.
Register all inputs and providers before evaluating a container; create a new
container when changing inputs or providers, since computed values are cached.

Classical T-L compensation:

```python

import matplotlib.pyplot as plt
from dafmit_aeromag import Dataset, Selection
from dataioc import DataIoC

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.foundation.sensors import MagVector, Tmi
from deinterf.metrics.fom import improve_rate, noise_level

if __name__ == "__main__":
    flt_d = Dataset().read(
        Selection(1002, lines="1002.02"),
        columns=["flux_b_x", "flux_b_y", "flux_b_z", "mag_3_uc"],
        split="train",
    )

    # prepare data
    tmi_with_interf = Tmi(tmi=flt_d["mag_3_uc"])
    fom_data = DataIoC().with_data(
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

This example uses the default IGRF14 model in `ppigrf>=2.1,<3`.
Its results can differ from earlier versions of the example that used IGRF13.

```python
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

    # date of flt1002
    year, doy = int(flt_d["year"].iloc[0]), int(flt_d["doy"].iloc[0])

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
