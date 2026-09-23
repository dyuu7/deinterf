import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import validate_params
from sklearn.utils.validation import check_array


@validate_params(
    {
        "magvec": ["array-like"],
        "copy": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def magvec2intensity(magvec: ArrayLike, copy=True) -> ndarray:
    """Compute magnetic field intensity from magnetic vectors.

    Parameters
    ----------
    magvec : array-like of shape (n_samples, 3)
        Magnetic field components in x, y, z order.
    copy : bool, default=True
        Copy the input during validation.

    Returns
    -------
    intensity : ndarray of shape (n_samples,)
        Vector magnitude for each sample, in the same units as the input.
    """
    magvec = check_array(magvec, ensure_min_features=3, copy=copy)
    return np.linalg.norm(magvec, axis=1)


@validate_params(
    {
        "magvec": ["array-like"],
        "copy": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def magvec2dircosine(magvec: ArrayLike, copy=True) -> ndarray:
    """Compute direction cosines from magnetic vectors.

    Parameters
    ----------
    magvec : array-like of shape (n_samples, 3)
        Magnetic field components in x, y, z order, with nonzero magnitudes.
    copy : bool, default=True
        Copy the input during validation.

    Returns
    -------
    dir_cosine : ndarray of shape (n_samples, 3)
        Unitless direction cosines in x, y, z order.
    """
    magvec = check_array(magvec, ensure_min_features=3, copy=copy)

    bx, by, bz = np.transpose(magvec)

    intensity = magvec2intensity(magvec)

    dir_conise_x = bx / intensity
    dir_conise_y = by / intensity
    dir_conise_z = bz / intensity

    return np.column_stack((dir_conise_x, dir_conise_y, dir_conise_z))
