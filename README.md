<h1 align="center">
  <img src="https://raw.githubusercontent.com/dyuu7/deinterf/main/docs/assets/logo.svg" alt="deinterf" width="80%" />
</h1>

Aeromagnetic interference compensation based on the
[Tolles–Lawson (T–L) model](https://en.wikipedia.org/wiki/Tolles%E2%80%93Lawson_equation).
`deinterf` estimates aircraft interference from magnetic-field measurements and
subtracts it from scalar total-field data. It provides a scikit-learn-style
`fit`/`predict`/`transform` API and composable model terms for adding other
interference sources.

The current package focuses on scalar TMI compensation with classical linear
T–L terms. It does not claim that a lower compensation metric always means a
more accurate geological field; validate a model on an independent flight or
line before using its output.

**English** | [简体中文](README.zh-CN.md)

## Install

`deinterf` supports CPython 3.10–3.14.

```shell
pip install deinterf
```

The core package accepts NumPy-compatible arrays. Install the optional reader
for the real-flight example below:

```shell
pip install "deinterf[examples]"
```

## Quick start

`MagVector` has shape `(n_samples, 3)` and `Tmi` has shape `(n_samples,)`.
Samples must be aligned, evenly sampled, and use consistent units and
coordinate conventions. Fit on calibration data, then transform an independent
flight or line:

```python
import numpy as np
from dataioc import DataIoC

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.foundation.sensors import MagVector, Tmi
from deinterf.metrics.fom import improve_rate


def flight(phase):
    time = np.arange(400) / 10  # 10 Hz
    field = np.column_stack(
        (
            20000 + 800 * np.sin(2 * np.pi * 0.17 * time + phase),
            3000 + 700 * np.cos(2 * np.pi * 0.23 * time + phase),
            45000 + 900 * np.sin(2 * np.pi * 0.31 * time + phase),
        )
    )
    direction = field / np.linalg.norm(field, axis=1)[:, None]
    interference = direction @ np.array([1200.0, -700.0, 300.0])
    return (
        DataIoC().with_data(MagVector(*field.T)),
        Tmi(50000 + interference),
    )


calibration_X, calibration_y = flight(0.0)
survey_X, survey_y = flight(0.7)

model = TollesLawson(terms=Terms.Terms_16, sampling_rate=10)
model.fit(calibration_X, calibration_y)
tmi_clean = model.transform(survey_X, survey_y)

print(f"improvement_rate={improve_rate(survey_y, tmi_clean):.2f}")
```

By default, `TollesLawson` standardizes features, fits a 10-fold `RidgeCV`,
and uses a 4th-order 0.1–0.6 Hz band-pass filter at the configured sampling
rate. Set `filter=None` when that preprocessing is not appropriate.

`predict(X)` returns the estimated interference with its constant component
removed. `transform(X, y)` returns the compensated TMI in the same units as
`y`; the mean level is preserved.

## Real-flight data

This example uses [`dafmit-aeromag`](https://github.com/dyuu7/dafmit-aeromag)
to read flight `1002`, calibration line `1002.02`, at 10 Hz. The reader may
download and verify the data on first use. Use its `offline=True` mode with an
existing cache when network access is unavailable.

```python
from dafmit_aeromag import Dataset, Selection
from dataioc import DataIoC

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.foundation.sensors import MagVector, Tmi
from deinterf.metrics.fom import improve_rate, noise_level


flight_data = Dataset().read(
    Selection(1002, lines="1002.02"),
    columns=["flux_b_x", "flux_b_y", "flux_b_z", "mag_3_uc"],
    split="train",
)

fom_data = DataIoC().with_data(
    MagVector(
        bx=flight_data["flux_b_x"],
        by=flight_data["flux_b_y"],
        bz=flight_data["flux_b_z"],
    )
)
tmi_with_interf = Tmi(flight_data["mag_3_uc"])

# This is a calibration-line demonstration. Use a separate line or flight
# for a generalization result.
tmi_clean = TollesLawson(terms=Terms.Terms_16).fit_transform(
    fom_data, tmi_with_interf
)
print(f"noise_level={noise_level(tmi_clean):.4f}")
print(f"improvement_rate={improve_rate(tmi_with_interf, tmi_clean):.2f}")
```

## Model terms and extension

| Term set | Features | Typical use |
| --- | ---: | --- |
| `Terms.Permanent` | 3 | permanent magnetization |
| `Terms.Induced_5` | 5 | induced magnetization |
| `Terms.Eddy_8` | 8 | eddy-current effects |
| `Terms.Terms_16` | 16 | standard T–L model; default |
| `Terms.Terms_18` | 18 | expanded induced and eddy terms |

The optional index selects a sensor source. For example,
`Terms.Terms_16[1]` consumes `MagVector[1]`; the unindexed form consumes the
default `MagVector` source.

Custom terms implement `ComposableTerm.__build__` and can be combined with
`|`, for example `Terms.Terms_16 | MyInterferenceTerm()`. A custom provider can
also replace a derived source such as `DirectionalCosine`. See
[`examples/extended_load_vibration_tmi.py`](examples/extended_load_vibration_tmi.py),
[`examples/extended_cable_current_tmi.py`](examples/extended_cable_current_tmi.py),
and [`examples/replace_direction_cosine_source_tmi.py`](examples/replace_direction_cosine_source_tmi.py).
Register all inputs and providers before evaluating a `DataIoC`; computed
values are cached by the container.

## Evaluation

`noise_level(y)` is the standard deviation after the 0.1–0.6 Hz band-pass
filter. `improve_rate(y_uncomp, y_comped)` is
`noise_level(y_uncomp) / noise_level(y_comped)`. Both default to 10 Hz; pass
the actual sampling rate when it differs.

Evaluate on data that was not used to fit the compensator. The example metrics
are useful for comparing processing choices, but they are not a substitute for
geophysical validation or a survey-specific quality-control procedure.

## Migration and development

Current releases use [`dataioc`](https://github.com/dyuu7/dataioc) for data
containers and `dafmit-aeromag` for the optional flight reader. The old
`deinterf.utils.data_ioc` and `sgl2020.Sgl2020` interfaces are no longer
supported.

From a checkout:

```shell
uv sync --extra examples
uv run --extra examples pytest
```

## Contributors

Thanks to everyone who contributes code, examples, bug reports, and feedback.

<a href="https://github.com/dyuu7/deinterf/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=dyuu7/deinterf" alt="deinterf contributors" />
</a>

The data container, now maintained as [`dataioc`](https://github.com/dyuu7/dataioc),
was primarily implemented by [yanang007](https://github.com/yanang007),
with the original idea proposed by [dyuu7](https://github.com/dyuu7).

## License

The project is licensed under the [MIT License](LICENSE).
