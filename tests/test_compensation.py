import numpy as np
import pytest
from dataioc import DataIoC
from sklearn.linear_model import LinearRegression

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.foundation.sensors import MagVector, Tmi
from deinterf.metrics.fom import improve_rate, noise_level


@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("sensor_id", [0, 1])
def test_compensation_recovers_known_interference_on_new_data(norm, sensor_id):
    def flight(seed):
        vector = np.random.default_rng(seed).normal(size=(400, 3))
        direction = vector / np.linalg.norm(vector, axis=1)[:, None]
        interference = direction @ np.array([12.0, -7.0, 3.0])
        container = DataIoC().with_data(MagVector[sensor_id](*(50000 * direction).T))
        return container, Tmi(50000 + interference), interference

    training, measured, _ = flight(0)
    model = TollesLawson(
        terms=Terms.Permanent[sensor_id],
        filter=None,
        norm=norm,
        estimator=LinearRegression(),
    )
    assert model.fit(training, measured) is model

    survey, measured, interference = flight(1)
    predicted = model.predict(survey)
    compensated = model.transform(survey, measured)
    assert isinstance(predicted, Tmi)
    assert isinstance(compensated, Tmi)
    assert compensated.shape == measured.shape
    np.testing.assert_allclose(predicted, interference - interference.mean(), atol=1e-9)
    np.testing.assert_allclose(compensated, measured.mean(), atol=1e-9)


def test_default_model_reduces_band_limited_interference_on_new_data():
    def flight(phase):
        time = np.arange(800) / 10
        vector = np.column_stack(
            (
                20000 + 800 * np.sin(2 * np.pi * 0.17 * time + phase),
                3000 + 700 * np.cos(2 * np.pi * 0.23 * time + phase),
                45000 + 900 * np.sin(2 * np.pi * 0.31 * time + phase),
            )
        )
        direction = vector / np.linalg.norm(vector, axis=1)[:, None]
        interference = direction @ np.array([1200.0, -700.0, 300.0])
        return DataIoC().with_data(MagVector(*vector.T)), Tmi(50000 + interference)

    training, measured = flight(0)
    model = TollesLawson().fit(training, measured)
    survey, measured = flight(0.7)
    predicted = model.predict(survey)
    compensated = model.transform(survey, measured)

    assert isinstance(predicted, Tmi)
    assert isinstance(compensated, Tmi)
    assert predicted.shape == compensated.shape == measured.shape
    assert np.isfinite(predicted).all()
    assert np.isfinite(compensated).all()
    np.testing.assert_allclose(compensated, measured - predicted)
    np.testing.assert_allclose(compensated.mean(), measured.mean(), atol=1e-9)
    assert noise_level(compensated) < noise_level(measured) / 10
    assert improve_rate(measured, compensated) > 10
