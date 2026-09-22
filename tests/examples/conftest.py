from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def local_flight(tmp_path: Path, monkeypatch):
    """Use the real dataset reader with a small offline calibration flight."""
    import h5py
    from dafmit_aeromag import Dataset

    samples = 400
    t = np.arange(samples) / 10
    bx = 20000 + 800 * np.sin(2 * np.pi * 0.17 * t)
    by = 3000 + 700 * np.cos(2 * np.pi * 0.23 * t)
    bz = 45000 + 900 * np.sin(2 * np.pi * 0.31 * t)
    fields = {
        "line": np.full(samples, 1002.02),
        "tt": 46380 + t,
        "mag_3_uc": np.sqrt(bx**2 + by**2 + bz**2),
        "mag_5_uc": np.sqrt(bx**2 + by**2 + bz**2) + np.sin(t),
        "ins_yaw": 10 * np.sin(t),
        "ins_pitch": 5 * np.sin(t / 2),
        "ins_roll": 5 * np.cos(t / 3),
        "lon": np.full(samples, -76.0),
        "lat": np.full(samples, 45.0),
        "utm_z": np.full(samples, 1000.0),
    }
    for sensor in ("b", "d"):
        fields.update(zip((f"flux_{sensor}_{axis}" for axis in "xyz"), (bx, by, bz)))
    for index, name in enumerate(
        (
            "com_1",
            "ac_hi",
            "ac_lo",
            "tank",
            "flap",
            "strb",
            "srvo_o",
            "srvo_m",
            "srvo_i",
            "heat",
            "acpwr",
            "outpwr",
            "bat_1",
            "bat_2",
        )
    ):
        fields[f"cur_{name}"] = 2 + np.sin(t * (index + 1) / 10)
    path = tmp_path / "Flt1002_train.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("N", data=samples)
        handle.create_dataset("dt", data=0.1)
        for name, values in fields.items():
            handle.create_dataset(name, data=values)

    def local_paths(self, flights):
        assert tuple(flights) == (1002,)
        return {1002: path}

    monkeypatch.setattr(Dataset, "fetch", local_paths)
    return Dataset(data_dir=tmp_path, offline=True, progress=False)
