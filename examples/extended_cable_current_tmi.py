import matplotlib.pyplot as plt
import numpy as np
from dafmit_aeromag import Dataset, Selection
from dataioc import DataIoC, DataNDArray
from numpy.typing import ArrayLike
from scipy.signal import butter, filtfilt

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.foundation import ComposableTerm
from deinterf.foundation.sensors import DirectionalCosine, MagVector, Tmi
from deinterf.metrics.fom import improve_rate


class Current(DataNDArray):
    def __new__(cls, current: ArrayLike, fs=10):
        b, a = butter(4, Wn=0.4, btype="low", fs=fs)
        current_filtered = filtfilt(b, a, current)
        return super().__new__(cls, current_filtered)


class Cable(ComposableTerm):
    def __build__(self, container: DataIoC):
        current = container[Current]
        dcos = container[DirectionalCosine[0]]
        ret1 = current[:, None] * dcos
        current_dot = np.gradient(current)
        ret2 = current_dot[:, None] * dcos
        return np.column_stack((ret1, ret2))


if __name__ == "__main__":
    # Load flight data.
    flt_d = Dataset().read(
        Selection(1002, lines="1002.02"),
        columns=[
            "flux_d_x",
            "flux_d_y",
            "flux_d_z",
            "mag_5_uc",
            "cur_com_1",
            "cur_ac_hi",
            "cur_ac_lo",
            "cur_tank",
            "cur_flap",
            "cur_strb",
            "cur_srvo_o",
            "cur_srvo_m",
            "cur_srvo_i",
            "cur_heat",
            "cur_acpwr",
            "cur_outpwr",
            "cur_bat_1",
            "cur_bat_2",
        ],
        split="train",
    )

    # Prepare input data.
    tmi_with_interf = Tmi(tmi=flt_d["mag_5_uc"])
    fom_data = DataIoC().with_data(
        MagVector(bx=flt_d["flux_d_x"], by=flt_d["flux_d_y"], bz=flt_d["flux_d_z"]),
        Current[0](current=flt_d["cur_com_1"]),
        Current[1](current=flt_d["cur_ac_hi"]),
        Current[2](current=flt_d["cur_ac_lo"]),
        Current[3](current=flt_d["cur_tank"]),
        Current[4](current=flt_d["cur_flap"]),
        Current[5](current=flt_d["cur_strb"]),
        Current[6](current=flt_d["cur_srvo_o"]),
        Current[7](current=flt_d["cur_srvo_m"]),
        Current[8](current=flt_d["cur_srvo_i"]),
        Current[9](current=flt_d["cur_heat"]),
        Current[10](current=flt_d["cur_acpwr"]),
        Current[11](current=flt_d["cur_outpwr"]),
        Current[12](current=flt_d["cur_bat_1"]),
        Current[13](current=flt_d["cur_bat_2"]),
    )

    # Extend the T-L terms with cable-current features.
    expanded_terms = (
        Terms.Terms_16
        | Cable()[0]
        | Cable()[1]
        | Cable()[2]
        | Cable()[3]
        | Cable()[4]
        | Cable()[5]
        | Cable()[6]
        | Cable()[7]
        | Cable()[8]
        | Cable()[9]
        | Cable()[10]
        | Cable()[11]
        | Cable()[12]
        | Cable()[13]
    )

    # Create a compensator with the extended terms.
    compensator = TollesLawson(terms=expanded_terms)
    # Fit and transform in one step.
    tmi_clean = compensator.fit_transform(fom_data, tmi_with_interf)

    # Evaluate compensation performance.
    ir = improve_rate(tmi_with_interf, tmi_clean, verbose=True)
    print(f"{ir=}")

    # Plot the input and compensated signals.
    plt.plot(tmi_with_interf, label="tmi_with_interf")
    plt.plot(tmi_clean, label="tmi_clean")
    plt.legend()
    plt.show()
