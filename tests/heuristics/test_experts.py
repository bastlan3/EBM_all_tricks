import pytest
import torch

from ebm_lib.heuristics.experts import LaplacianVarianceExpert, HighFrequencyEnergyExpert

@pytest.fixture
def image_batch():
    """Provides a standard batch of images for testing."""
    return torch.randn(4, 3, 32, 32) # Batch of 4, 3-channel, 32x32 images

def test_laplacian_variance_expert(image_batch):
    """Tests the LaplacianVarianceExpert."""
    expert = LaplacianVarianceExpert()
    scores = expert(image_batch)

    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (image_batch.shape[0],) # Should be (batch_size,)

def test_high_frequency_energy_expert(image_batch):
    """Tests the HighFrequencyEnergyExpert."""
    expert = HighFrequencyEnergyExpert()
    scores = expert(image_batch)

    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (image_batch.shape[0],) # Should be (batch_size,)
    # The score should be a ratio, so between 0 and 1
    assert torch.all(scores >= 0) and torch.all(scores <= 1)

def test_experts_differentiate_noise():
    """
    A functional test to ensure experts produce different scores for a
    blank image versus a noisy image.
    """
    blank_image = torch.zeros(1, 1, 32, 32)
    noisy_image = torch.rand(1, 1, 32, 32) * 0.5 + 0.25 # Random noise

    # Test Laplacian Variance Expert
    laplacian_expert = LaplacianVarianceExpert()
    blank_laplacian_score = laplacian_expert(blank_image)
    noisy_laplacian_score = laplacian_expert(noisy_image)

    # Noisy image should have a significantly higher variance of laplacian
    assert noisy_laplacian_score > blank_laplacian_score
    # Blank image score should be close to zero
    assert torch.isclose(blank_laplacian_score, torch.tensor(0.0))

    # Test High Frequency Energy Expert
    fft_expert = HighFrequencyEnergyExpert()
    blank_fft_score = fft_expert(blank_image)
    noisy_fft_score = fft_expert(noisy_image)

    # Noisy image should have much more high-frequency energy
    assert noisy_fft_score > blank_fft_score
    # Blank image score should be close to zero
    assert torch.isclose(blank_fft_score, torch.tensor(0.0), atol=1e-6)
