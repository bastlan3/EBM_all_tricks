import pytest
import torch

from ebm_lib.estimators.knn_entropy import KnnEntropyEstimator

def test_knn_estimator_init():
    """Tests the initialization of the KnnEntropyEstimator."""
    estimator = KnnEntropyEstimator(k=3)
    assert estimator.k == 3

    with pytest.raises(ValueError):
        KnnEntropyEstimator(k=0)

def test_knn_estimator_forward_shape():
    """Tests that the estimator returns a scalar tensor."""
    estimator = KnnEntropyEstimator(k=5)
    samples = torch.randn(20, 10) # 20 samples, 10 dimensions
    loss = estimator(samples)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

def test_knn_estimator_entropy_logic():
    """
    Tests that the entropy proxy is lower for clustered points (low entropy)
    and higher for spread-out points (high entropy). Since our loss is the
    negative log distance, a lower loss means higher entropy.
    """
    estimator = KnnEntropyEstimator(k=3)

    # Batch 1: Tightly clustered points
    clustered_samples = torch.randn(10, 2) * 0.1

    # Batch 2: Widely spread-out points
    spread_samples = torch.randn(10, 2) * 10.0

    clustered_loss = estimator(clustered_samples)
    spread_loss = estimator(spread_samples)

    # The loss for the spread-out samples (higher entropy) should be lower
    # because the distances to neighbors are larger.
    assert spread_loss < clustered_loss

def test_knn_estimator_handles_insufficient_samples():
    """
    Tests that the estimator returns zero loss if the number of samples
    is less than or equal to k.
    """
    estimator = KnnEntropyEstimator(k=5)
    samples = torch.randn(4, 10) # Only 4 samples, k=5
    loss = estimator(samples)

    assert torch.isclose(loss, torch.tensor(0.0))