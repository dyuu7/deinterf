from __future__ import annotations

from typing import Iterable

import numpy as np
from dataioc import DataDescriptor, DataIoC


class ComposableTerm(DataDescriptor[np.ndarray]):
    def __or__(self, other: ComposableTerm):
        return Composition(self, other)


class Composition(ComposableTerm):
    __slots__ = ['terms']

    def __init__(
            self,
            terms: ComposableTerm | Composition | Iterable[ComposableTerm],
            *other_terms: ComposableTerm | Composition | Iterable[ComposableTerm],
            **kwargs
    ):
        super().__init__(**kwargs)

        _terms: list[ComposableTerm] = []
        for term in [terms, *other_terms]:
            if isinstance(term, Iterable):
                _terms.extend(term)
            else:
                _terms.append(term)

        self.terms = tuple(_terms)

    def __getitem__(self, item):
        return type(self)(term.index_implicit(item) for term in self.terms)

    def __iter__(self):
        yield from self.terms

    def __build__(self, container: DataIoC):
        return np.column_stack([container[term] for term in self.terms])
