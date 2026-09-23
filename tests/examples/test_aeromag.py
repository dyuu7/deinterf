import re
import runpy
from pathlib import Path

import numpy as np
import pytest
from dataioc import DataIoC

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.foundation.sensors import DirectionalCosine, MagVector, Tmi

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")
Selection = pytest.importorskip("dafmit_aeromag").Selection
pytest.importorskip("ppigrf")


def test_dataset_dataframe_builds_indexed_compensation_features(local_flight):
    frame = local_flight.read(
        Selection(1002, lines="1002.02"),
        columns=["flux_b_x", "flux_b_y", "flux_b_z", "mag_3_uc"],
        split="train",
    )
    assert frame["line"].unique().tolist() == ["1002.02"]
    assert frame["year"].iloc[0] == 2020
    assert frame["doy"].iloc[0] == 172
    container = DataIoC().with_data(
        MagVector[1](frame["flux_b_x"], frame["flux_b_y"], frame["flux_b_z"])
    )
    features = container[Terms.Terms_16[1]]
    assert features.shape == (400, 16)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(
        np.linalg.norm(container[DirectionalCosine[1]], axis=1), 1
    )
    result = TollesLawson(terms=Terms.Terms_16[1]).fit_transform(
        container, Tmi(frame["mag_3_uc"])
    )
    assert result.shape == (400,)
    assert np.isfinite(result).all()


@pytest.mark.parametrize(
    "example",
    [
        "classic_tmi.py",
        "extended_load_vibration_tmi.py",
        "extended_cable_current_tmi.py",
        "replace_direction_cosine_source_tmi.py",
    ],
)
def test_examples_run_with_real_reader_and_local_flight(
    example, local_flight, monkeypatch
):
    monkeypatch.setattr(plt, "show", lambda: None)
    path = Path(__file__).parents[2] / "examples" / example
    try:
        namespace = runpy.run_path(str(path), run_name="__main__")
        result_names = (
            ("tmi_clean_classic", "tmi_clean_ins")
            if "tmi_clean_ins" in namespace
            else ("tmi_clean",)
        )
        for name in result_names:
            result = namespace[name]
            assert isinstance(result, np.ndarray)
            assert result.shape == (400,)
            assert np.isfinite(result).all()
    finally:
        plt.close("all")


@pytest.mark.parametrize("index", [0, 1])
def test_readme_examples_run_with_local_flight(index, local_flight, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    readme = Path(__file__).parents[2] / "README.md"
    blocks = re.findall(r"```python\n(.*?)```", readme.read_text(), re.DOTALL)
    namespace = {"__name__": "__main__"}
    try:
        exec(compile(blocks[index], str(readme), "exec"), namespace)
        result = namespace.get("tmi_clean", namespace.get("tmi_clean_ins"))
        assert isinstance(result, np.ndarray)
        assert result.shape == (400,)
        assert np.isfinite(result).all()
    finally:
        plt.close("all")
