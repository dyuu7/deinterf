import matplotlib.pyplot as plt
from dafmit_aeromag import Dataset, Selection
from dataioc import DataIoC

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.foundation.sensors import MagVector, Tmi
from deinterf.metrics.fom import improve_rate, noise_level

if __name__ == "__main__":
    # Load flight data.
    flt_d = Dataset().read(
        Selection(1002, lines="1002.02"),
        columns=["flux_b_x", "flux_b_y", "flux_b_z", "mag_3_uc"],
        split="train",
    )

    # Prepare input data.
    tmi_with_interf = Tmi(tmi=flt_d["mag_3_uc"])
    fom_data = DataIoC().with_data(
        MagVector(bx=flt_d["flux_b_x"], by=flt_d["flux_b_y"], bz=flt_d["flux_b_z"])
    )

    # Create a compensator.
    compensator = TollesLawson(terms=Terms.Terms_16)
    # The default estimator uses cross-validated ridge regression. Replace it with
    # another regressor when needed.
    # from sklearn.linear_model import BayesianRidge
    # compensator = TollesLawson(terms=Terms.Terms_16, estimator=BayesianRidge())
    # Fit the compensator.
    compensator.fit(fom_data, tmi_with_interf)

    # Transform the signal with the fitted compensator.
    tmi_clean = compensator.transform(fom_data, tmi_with_interf)

    # Fit and transform in one step.
    tmi_clean = compensator.fit_transform(fom_data, tmi_with_interf)

    # Predict magnetic interference only.
    interf = compensator.predict(fom_data)

    # Evaluate compensation performance.
    comped_noise_level = noise_level(tmi_clean)
    print(f"{comped_noise_level=}")

    ir = improve_rate(tmi_with_interf, tmi_clean)
    print(f"{ir=}")

    # Plot the input and compensated signals.
    plt.plot(tmi_with_interf, label="tmi_with_interf")
    plt.plot(tmi_clean, label="tmi_clean")
    plt.legend()
    plt.show()
