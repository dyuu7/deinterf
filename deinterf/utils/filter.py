from numbers import Integral

from scipy.signal import butter, filtfilt
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.validation import check_array


@validate_params(
    {
        "X": ["array-like"],
        "bandpass_range": [tuple],
        "sampling_rate": [Interval(Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def fom_bpfilter(X, bandpass_range=(0.1, 0.6), sampling_rate=10):
    """Apply a bandpass filter to figure-of-merit (FOM) flight signals.

    Use a Butterworth filter with forward and backward passes along the sample
    axis to avoid phase shifts.

    Parameters
    ----------
    X : array-like of shape (n_samples,) or (n_samples, n_features)
        Signals to filter, with samples along the first axis.
    bandpass_range : tuple of float, default=(0.1, 0.6)
        Lower and upper cutoff frequencies in Hz.
    sampling_rate : int, default=10
        Sampling rate in Hz.

    Returns
    -------
    filtered : ndarray of shape (n_samples,) or (n_samples, n_features)
        Filtered signals with the same shape and units as the input.
    """
    X = check_array(X, copy=True, ensure_2d=False)
    b, a = butter(
        4,
        Wn=list(bandpass_range),
        btype="bandpass",
        fs=sampling_rate,
        output="ba",
    )
    filtered = filtfilt(b, a, X, axis=0)
    return filtered
