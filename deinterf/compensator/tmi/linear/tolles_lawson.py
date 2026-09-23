from __future__ import annotations

import sys
from numbers import Integral
from typing import Literal

import numpy as np
from scipy.signal import detrend
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, _fit_context
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.validation import check_consistent_length, check_is_fitted

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from dataioc import DataIoC

from deinterf.compensator.tmi.linear.terms import Terms
from deinterf.foundation import ComposableTerm, Composition
from deinterf.foundation.sensors import Tmi
from deinterf.utils.filter import fom_bpfilter


class TollesLawson(OneToOneFeatureMixin, BaseEstimator):
    """Compensate TMI signals with the Tolles-Lawson model.

    Build features from a ``DataIoC`` container and fit a regressor to estimate
    magnetic interference. Compensation subtracts the predicted interference
    while preserving the mean of the measured signal.

    Parameters
    ----------
    filter : {"bandpass", None}, default="bandpass"
        Filter features and measurements between 0.1 and 0.6 Hz during fitting.
        Use None to fit without filtering.
    terms : ComposableTerm or Composition, default=Terms.Terms_16
        Model terms used to build the feature matrix.
    norm : bool, default=True
        Center and scale features before filtering and fitting. Apply the fitted
        scaler to features during prediction.
    sampling_rate : int, default=10
        Sampling rate in Hz.
    estimator : object with fit and predict methods or None, default=None
        Regressor used to estimate magnetic interference. None selects ridge
        regression without an intercept, using 10-fold cross-validation over
        13 regularization strengths from 1e-6 to 1e6.
    copy : bool, default=True
        Stored parameter with no effect on data copying in this implementation.

    Attributes
    ----------
    model_ : estimator
        Fitted regressor.
    scaler_ : StandardScaler
        Feature scaler fitted when ``norm=True``.
    """

    _estimator_type = "regressor"
    _parameter_constraints: dict = {
        "filter": [StrOptions({"bandpass"}), None],
        "terms": [ComposableTerm, Composition],
        "norm": ["boolean"],
        "sampling_rate": [Interval(Integral, 1, None, closed="left")],
        "estimator": [HasMethods(["fit", "predict"]), None],
        "copy": ["boolean"],
    }

    def __init__(
            self,
            filter: Literal["bandpass"] | None = "bandpass",
            terms=Terms.Terms_16,
            norm=True,
            sampling_rate=10,
            estimator=None,
            copy=True,
    ) -> None:
        self.filter = filter
        self.terms = terms
        self.norm = norm
        self.sampling_rate = sampling_rate
        self.estimator = RidgeCV(
            fit_intercept=False,
            alphas=np.logspace(-6, 6, 13),
            cv=10,
        ) if estimator is None else estimator
        self.copy = copy

    def _reset(self) -> None:
        if hasattr(self, "model_"):
            del self.model_
        if hasattr(self, "scaler_"):
            del self.scaler_

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: DataIoC, y: Tmi) -> Self:
        """Clear fitted attributes and fit the compensation model.

        Parameters
        ----------
        X : DataIoC
            Container providing the data needed by the model terms.
        y : Tmi of shape (n_samples,)
            Uncompensated TMI measurements aligned with the samples in X.

        Returns
        -------
        self : TollesLawson
            Fitted compensator.
        """
        self._reset()
        return self.partial_fit(X, y)

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X: DataIoC, y: Tmi) -> Self:
        """Fit the compensation model without clearing fitted attributes first.

        Each call fits the feature scaler when normalization is enabled and
        calls the regressor's ``fit`` method. This method does not accumulate
        samples or call the regressor's ``partial_fit`` method.

        Parameters
        ----------
        X : DataIoC
            Container providing the data needed by the model terms.
        y : Tmi of shape (n_samples,)
            Uncompensated TMI measurements aligned with the samples in X.

        Returns
        -------
        self : TollesLawson
            Fitted compensator.
        """
        measurement = y
        tl_features = X[self.terms]
        check_consistent_length(tl_features, measurement)

        if self.norm:
            self.scaler_ = StandardScaler()
            tl_features = self.scaler_.fit_transform(tl_features)

        # Filter both sides to fit bpf(A) x = bpf(b), or fit A x = b directly.
        tl_features = (
            fom_bpfilter(tl_features, sampling_rate=self.sampling_rate)
            if self.filter == "bandpass"
            else tl_features
        )

        interf_measured = (
            fom_bpfilter(measurement, sampling_rate=self.sampling_rate)
            if self.filter == "bandpass"
            else measurement
        )

        self.model_ = self.estimator.fit(tl_features, interf_measured)

        return self

    def transform(self, X: DataIoC, y: Tmi) -> Tmi:
        """Subtract predicted magnetic interference from TMI measurements.

        Parameters
        ----------
        X : DataIoC
            Container providing the data needed by the model terms.
        y : Tmi of shape (n_samples,)
            Uncompensated TMI measurements aligned with the samples in X.

        Returns
        -------
        comped : Tmi of shape (n_samples,)
            Compensated measurements, preserving the input mean and units.
        """
        check_is_fitted(self)
        measurement = y
        interf = self.predict(X)
        comped = measurement - interf

        return Tmi(tmi=comped)

    def fit_transform(self, X: DataIoC, y: Tmi) -> Tmi:
        """Fit the compensation model and compensate the same measurements.

        Parameters
        ----------
        X : DataIoC
            Container providing the data needed by the model terms.
        y : Tmi of shape (n_samples,)
            Uncompensated TMI measurements aligned with the samples in X.

        Returns
        -------
        comped : Tmi of shape (n_samples,)
            Compensated measurements, preserving the input mean and units.
        """
        return self.fit(X, y).transform(X, y)

    def predict(self, X: DataIoC) -> Tmi:
        """Predict magnetic interference with its mean removed.

        Use features without bandpass filtering and apply the fitted scaler
        when normalization is enabled. Remove the mean of the predictions
        over the supplied samples.

        Parameters
        ----------
        X : DataIoC
            Container providing the data needed by the model terms.

        Returns
        -------
        interf : Tmi of shape (n_samples,)
            Predicted magnetic interference in the units of the training TMI.
        """
        check_is_fitted(self)

        tl_feats = X[self.terms]
        if self.norm:
            tl_feats = self.scaler_.transform(tl_feats)

        pred = self.model_.predict(tl_feats)
        interf = detrend(pred, axis=0, type="constant")

        return Tmi(tmi=interf)

    def fit_predict(self, X: DataIoC, y: Tmi) -> Tmi:
        """Fit the compensation model and predict interference for the same data.

        Parameters
        ----------
        X : DataIoC
            Container providing the data needed by the model terms.
        y : Tmi of shape (n_samples,)
            Uncompensated TMI measurements aligned with the samples in X.

        Returns
        -------
        interf : Tmi of shape (n_samples,)
            Predicted magnetic interference with its mean removed, in the
            same units as y.
        """
        return self.fit(X, y).predict(X)

    def _more_tags(self):
        return {
            "requires_y": True,
        }
