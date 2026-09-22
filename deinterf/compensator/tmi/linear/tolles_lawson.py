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
        self._reset()
        return self.partial_fit(X, y)

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X: DataIoC, y: Tmi) -> Self:
        measurement = y
        tl_features = X[self.terms]
        check_consistent_length(tl_features, measurement)

        if self.norm:
            self.scaler_ = StandardScaler()
            tl_features = self.scaler_.fit_transform(tl_features)

        # 决定是计算Ax=b形式，还是bpf(Ax)=bpf(b)=>bpf(A)x=bpf(b)形式的模型
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
        check_is_fitted(self)
        measurement = y
        interf = self.predict(X)
        comped = measurement - interf

        return Tmi(tmi=comped)

    def fit_transform(self, X: DataIoC, y: Tmi) -> Tmi:
        return self.fit(X, y).transform(X, y)

    def predict(self, X: DataIoC) -> Tmi:
        check_is_fitted(self)

        tl_feats = X[self.terms]
        if self.norm:
            tl_feats = self.scaler_.transform(tl_feats)

        pred = self.model_.predict(tl_feats)
        interf = detrend(pred, axis=0, type="constant")

        return Tmi(tmi=interf)

    def fit_predict(self, X: DataIoC, y: Tmi) -> Tmi:
        return self.fit(X, y).predict(X)

    def _more_tags(self):
        return {
            "requires_y": True,
        }
