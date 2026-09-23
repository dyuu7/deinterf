from numbers import Integral

import numpy as np
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.validation import check_consistent_length, column_or_1d

from deinterf.utils.filter import fom_bpfilter


@validate_params(
    {
        "y": ["array-like"],
        "sampling_rate": [Interval(Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def noise_level(y, sampling_rate=10):
    """Compute the noise level of a figure-of-merit (FOM) flight signal.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Magnetic field measurements.
    sampling_rate : int, default=10
        Sampling rate in Hz.

    Returns
    -------
    noise_level : float
        Standard deviation after filtering between 0.1 and 0.6 Hz, in the
        same units as the input.
    """
    y = column_or_1d(y, dtype=np.float64)
    filtered = fom_bpfilter(y, sampling_rate=sampling_rate)
    noise_level = np.std(filtered)
    return noise_level


@validate_params(
    {
        "y_uncomp": ["array-like"],
        "y_comped": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def improve_rate(y_uncomp, y_comped, sampling_rate=10, verbose=False):
    """Compute the compensation improvement ratio for a FOM flight signal.

    Parameters
    ----------
    y_uncomp : array-like of shape (n_samples,)
        Magnetic field measurements before compensation.
    y_comped : array-like of shape (n_samples,)
        Magnetic field measurements after compensation.
    sampling_rate : int, default=10
        Sampling rate in Hz.
    verbose : bool, default=False
        Print the noise levels before and after compensation.

    Returns
    -------
    ir : float
        Noise level before compensation divided by the noise level after
        compensation. Values greater than one indicate reduced noise.
    """
    check_consistent_length(y_uncomp, y_comped)
    y_uncomp = column_or_1d(y_uncomp, dtype=np.float64)
    y_comped = column_or_1d(y_comped, dtype=np.float64)

    uncomped_noise_level = noise_level(y_uncomp, sampling_rate)
    comped_noise_level = noise_level(y_comped, sampling_rate)
    if verbose:
        print(f"uncomped noise level: {uncomped_noise_level:.4f}")
        print(f"comped noise level: {comped_noise_level:.4f}")
    ir = uncomped_noise_level / comped_noise_level
    return ir
