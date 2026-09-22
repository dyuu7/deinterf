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

    # Data preparation
    tmi_with_interf = Tmi(tmi=flt_d["mag_3_uc"])
    fom_data = DataIoC().with_data(
        MagVector(bx=flt_d["flux_b_x"], by=flt_d["flux_b_y"], bz=flt_d["flux_b_z"])
    )

    # Create compensator
    compensator = TollesLawson(terms=Terms.Terms_16)
    # Default uses cross-validated ridge regression, can be replaced with other regressors
    # from sklearn.linear_model import BayesianRidge
    # compensator = TollesLawson(terms=Terms.Terms_16, estimator=BayesianRidge())
    compensator.fit(fom_data, tmi_with_interf)

    # Compensate given signal
    tmi_clean = compensator.transform(fom_data, tmi_with_interf)

    # Or fit and transform in one step
    tmi_clean = compensator.fit_transform(fom_data, tmi_with_interf)

    # Only predict magnetic interference
    interf = compensator.predict(fom_data)

    # Evaluate magnetic compensation performance
    comped_noise_level = noise_level(tmi_clean)
    print(f"{comped_noise_level=}")

    ir = improve_rate(tmi_with_interf, tmi_clean)
    print(f"{ir=}")

    # Simple plot
    plt.plot(tmi_with_interf, label="tmi_with_interf")
    plt.plot(tmi_clean, label="tmi_clean")
    plt.legend()
    plt.show()
