import matplotlib.pyplot as plt
import numpy as np
from dafmit_aeromag import Dataset, Selection
from dataioc import DataIoC

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.foundation import ComposableTerm
from deinterf.foundation.sensors import DirectionalCosine, MagIntensity, MagVector, Tmi
from deinterf.metrics.fom import improve_rate


class LoadVibration(ComposableTerm):
    def __build__(self, container: DataIoC) -> np.ndarray:
        intensity = container[MagIntensity]
        direction_cos = container[DirectionalCosine]
        dcx, dcy, dcz = direction_cos.T

        dcx_derivative = np.gradient(dcx)
        dcy_derivative = np.gradient(dcy)
        dcz_derivative = np.gradient(dcz)

        # Build third-order direction-cosine terms (10 features).
        triple_direction_terms = intensity[:, None] * np.column_stack(
            (
                dcx * dcx * dcx,
                dcx * dcx * dcy,
                dcx * dcx * dcz,
                dcx * dcy * dcy,
                dcx * dcy * dcz,
                dcx * dcz * dcz,
                dcy * dcy * dcy,
                dcy * dcy * dcz,
                dcy * dcz * dcz,
                dcz * dcz * dcz,
            )
        )

        # Build second-order direction-cosine terms with derivatives (18 features).
        double_direction_derivative_terms = intensity[:, None] * np.column_stack(
            (
                dcx * dcx * dcx_derivative,
                dcx * dcx * dcy_derivative,
                dcx * dcx * dcz_derivative,
                dcx * dcy * dcx_derivative,
                dcx * dcy * dcy_derivative,
                dcx * dcy * dcz_derivative,
                dcx * dcz * dcx_derivative,
                dcx * dcz * dcy_derivative,
                dcx * dcz * dcz_derivative,
                dcy * dcy * dcx_derivative,
                dcy * dcy * dcy_derivative,
                dcy * dcy * dcz_derivative,
                dcy * dcz * dcx_derivative,
                dcy * dcz * dcy_derivative,
                dcy * dcz * dcz_derivative,
                dcz * dcz * dcx_derivative,
                dcz * dcz * dcy_derivative,
                dcz * dcz * dcz_derivative,
            )
        )

        # Build first-order derivative terms (3 features).
        derivative_terms = intensity[:, None] * np.column_stack(
            (
                dcx_derivative,
                dcy_derivative,
                dcz_derivative,
            )
        )

        return np.column_stack(
            (
                triple_direction_terms,
                double_direction_derivative_terms,
                derivative_terms,
            )
        )


if __name__ == "__main__":
    # Load flight data.
    flt_d = Dataset().read(
        Selection(1002, lines="1002.02"),
        columns=["flux_d_x", "flux_d_y", "flux_d_z", "mag_5_uc"],
        split="train",
    )

    # Prepare input data.
    tmi_with_interf = Tmi(tmi=flt_d["mag_5_uc"])
    fom_data = DataIoC().with_data(
        MagVector(bx=flt_d["flux_d_x"], by=flt_d["flux_d_y"], bz=flt_d["flux_d_z"])
    )

    # Create a compensator with the extended terms.
    compensator = TollesLawson(terms=Terms.Terms_16 | LoadVibration())

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
