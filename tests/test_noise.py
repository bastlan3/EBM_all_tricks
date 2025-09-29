import pytest
import torch
from ebm_lib.noise import NoiseGenerator

@pytest.fixture
def noise_gen():
    """Provides a NoiseGenerator instance."""
    return NoiseGenerator(size=(64, 64), batch_size=4, device=torch.device("cpu"))

def test_noise_generator_shape(noise_gen):
    """Tests that all noise generation methods produce the correct shape."""
    for noise_func in [
        noise_gen.generate_white_noise,
        noise_gen.generate_pink_noise,
        noise_gen.generate_brown_noise,
    ]:
        noise = noise_func()
        assert noise.shape == (4, 1, 64, 64)

def test_curriculum_noise_shape(noise_gen):
    """Tests the curriculum noise generation."""
    noise = noise_gen.generate_curriculum_noise(progress=0.5)
    assert noise.shape == (4, 1, 64, 64)
    with pytest.raises(ValueError):
        noise_gen.generate_curriculum_noise(progress=1.1)

def _get_spectrum_slope(noise_tensor):
    """Helper function to calculate the slope of the log-log power spectrum."""
    # Average over the batch and channels
    noise_fft = torch.fft.fft2(noise_tensor.mean(dim=(0, 1)))
    power_spectrum = torch.abs(noise_fft)**2

    # Calculate radial frequency
    h, w = power_spectrum.shape
    freqs_h = torch.fft.fftfreq(h)
    freqs_w = torch.fft.fftfreq(w)
    radius = torch.sqrt(torch.square(freqs_h[:, None]) + torch.square(freqs_w[None, :]))

    # Bin the frequencies and power spectrum
    num_bins = 20
    bins = torch.linspace(0, radius.max(), num_bins + 1)
    binned_power = torch.zeros(num_bins)
    bin_counts = torch.zeros(num_bins)

    for i in range(num_bins):
        mask = (radius >= bins[i]) & (radius < bins[i+1])
        if mask.any():
            binned_power[i] = power_spectrum[mask].mean()
            bin_counts[i] = 1

    # Filter out empty bins and take logs
    valid_bins = binned_power > 0
    log_freq = torch.log(bins[1:][valid_bins])
    log_power = torch.log(binned_power[valid_bins])

    # Fit a line to the log-log plot to find the slope
    A = torch.vstack([log_freq, torch.ones_like(log_freq)]).T
    # Use the modern torch.linalg.lstsq which has reversed arguments
    solution = torch.linalg.lstsq(A, log_power.unsqueeze(1)).solution

    return solution[0].item()

@pytest.mark.parametrize("noise_type, expected_slope", [
    ("white", 0.0),
    ("pink", -1.0),
    ("brown", -2.0)
])
def test_noise_spectrum(noise_gen, noise_type, expected_slope):
    """
    Tests that the generated noise has the correct spectral properties by
    checking the slope of its log-log power spectrum.
    """
    if noise_type == "white":
        noise = noise_gen.generate_white_noise()
    elif noise_type == "pink":
        noise = noise_gen.generate_pink_noise()
    elif noise_type == "brown":
        noise = noise_gen.generate_brown_noise()

    slope = _get_spectrum_slope(noise)

    # We check if the calculated slope is reasonably close to the expected value.
    assert abs(slope - expected_slope) < 0.5