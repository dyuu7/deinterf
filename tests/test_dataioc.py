import numpy as np
import pytest
from dataioc import DataIoC, DataNDArray, UniqueData

from deinterf.compensator.tmi.linear import Terms
from deinterf.foundation import ComposableTerm
from deinterf.foundation.sensors import DirectionalCosine, MagIntensity, MagVector


@pytest.mark.parametrize("terms, columns", [(Terms.Terms_16, 16), (Terms.Terms_18, 18)])
def test_indexed_sensors_build_and_cache_compensation_terms(terms, columns):
    container = DataIoC().with_data(
        MagVector([4.0, 4.0, 4.0], [0.0, 3.0, 0.0], [3.0, 0.0, -3.0]),
        MagVector[1]([3.0, 0.0, -3.0], [0.0, 3.0, 0.0], [4.0, 4.0, 4.0]),
    )
    expected = [[0.6, 0, 0.8], [0, 0.6, 0.8], [-0.6, 0, 0.8]]
    direction = container[DirectionalCosine[1]]
    assert isinstance(direction, DataNDArray)
    np.testing.assert_allclose(direction, expected)
    np.testing.assert_allclose(container[MagIntensity[1]], [5, 5, 5])

    features = container[terms[1]]
    assert features.shape == (3, columns)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features[:, :3], expected)
    assert container[terms[1]] is features
    assert not np.allclose(container[terms], features)


def test_composition_preserves_explicit_sensor_references():
    container = DataIoC().with_data(
        MagVector([3.0, 0.0], [0.0, 3.0], [4.0, 4.0]),
        MagVector[1]([0.0, 4.0], [4.0, 0.0], [3.0, 3.0]),
    )
    terms = (Terms.Permanent | Terms.Permanent[0])[1]
    np.testing.assert_allclose(
        container[terms],
        [[0, 0.8, 0.6, 0.6, 0, 0.8], [0.8, 0, 0.6, 0, 0.6, 0.8]],
    )


def test_custom_term_shares_unique_data_across_sensor_ids():
    class SharedField(DataNDArray, UniqueData):
        pass

    class SharedTerm(ComposableTerm):
        def __build__(self, container):
            return container[SharedField]

    shared = SharedField([1.0, 2.0])
    container = DataIoC().with_data(shared)
    assert container[SharedTerm[0]] is shared
    assert container[SharedTerm[1]] is shared


def test_provider_override_remaps_sensor_id():
    class Replacement(DirectionalCosine):
        @classmethod
        def __build__(cls, container):
            vector = container[MagVector]
            rotated = vector[:, [1, 0, 2]]
            return cls(*(rotated / np.linalg.norm(rotated, axis=1)[:, None]).T)

    container = DataIoC(record_all=True).with_data(
        MagVector([4.0, 4.0], [0.0, 0.0], [3.0, 3.0]),
        MagVector[1]([3.0, 0.0], [0.0, 3.0], [4.0, 4.0]),
    )
    container.add_provider(DirectionalCosine, Replacement)
    result = container[DirectionalCosine[1]]
    np.testing.assert_allclose(result, [[0, 0.6, 0.8], [0.6, 0, 0.8]])
    np.testing.assert_allclose(container[Terms.Terms_16[1]][:, :3], result)
    assert "Replacement" in str(container.logger)
    assert "MagVector[1]" in str(container.logger)
