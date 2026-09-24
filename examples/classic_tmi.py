"""Fit on calibration line 1002.02 and compensate repeat line 158.00.

Run with: uv run --extra examples python examples/classic_tmi.py
"""

import matplotlib.pyplot as plt
from dafmit_aeromag import Dataset, Selection
from dataioc import DataIoC

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.foundation.sensors import MagVector, Tmi
from deinterf.metrics.fom import improve_rate


if __name__ == "__main__":
    dataset = Dataset()
    columns = ["flux_b_x", "flux_b_y", "flux_b_z", "mag_3_uc"]
    calibration = dataset.read(
        Selection(1002, lines="1002.02"), columns=columns, split="train"
    )
    survey = dataset.read(
        Selection(1002, lines="158.00"), columns=columns, split="train"
    )

    def as_inputs(frame):
        X = DataIoC().with_data(
            MagVector(frame["flux_b_x"], frame["flux_b_y"], frame["flux_b_z"])
        )
        return X, Tmi(frame["mag_3_uc"])

    calibration_X, calibration_y = as_inputs(calibration)
    survey_X, survey_y = as_inputs(survey)

    model = TollesLawson(terms=Terms.Terms_16, sampling_rate=10)
    model.fit(calibration_X, calibration_y)
    tmi_clean = model.transform(survey_X, survey_y)
    print(f"improvement_rate={improve_rate(survey_y, tmi_clean, sampling_rate=10):.2f}")

    plt.plot(survey_y, label="uncompensated")
    plt.plot(tmi_clean, label="compensated")
    plt.xlabel("Sample")
    plt.ylabel("TMI (nT)")
    plt.legend()
    plt.show()
