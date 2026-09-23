from __future__ import annotations

from typing import Iterable

import numpy as np
from dataioc import DataDescriptor, DataIoC


class ComposableTerm(DataDescriptor[np.ndarray]):
    """Feature descriptor that supports composition and data indexing.

    Implement ``__build__`` to resolve dependencies from a ``DataIoC`` container
    and return feature columns. Use ``|`` to combine terms and ``[index]`` to
    select a data source.
    """

    def __or__(self, other: ComposableTerm):
        """Combine terms in the order of their feature columns."""
        return Composition(self, other)

    def __getitem__(self, index) -> ComposableTerm:
        """Return a term bound to an explicit data index."""
        return self.index_explicit(index)


class Composition(ComposableTerm):
    """Ordered collection of terms whose feature columns are concatenated.

    Terms may be passed individually or as iterables, including other
    compositions. Indexing a composition applies the index to terms without
    an explicit index, preserving any explicit data indices.
    """

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

    def __getitem__(self, index):
        """Apply a data index while preserving explicit term indices."""
        return type(self)(term.index_implicit(index) for term in self.terms)

    def __iter__(self):
        """Iterate over terms in the order of their feature columns."""
        yield from self.terms

    def __build__(self, container: DataIoC):
        """Resolve each term and concatenate its feature columns."""
        return np.column_stack([container[term] for term in self.terms])
