import torch

class NoiseGenerator:
    """
    A class for generating different "colors" of noise by shaping the
    spectrum of random noise in the frequency domain.
    """
    def __init__(self, size: tuple, batch_size: int, device: torch.device):
        """
        Args:
            size (tuple): The spatial dimensions of the noise (e.g., (H, W)).
            batch_size (int): The number of noise samples to generate.
            device (torch.device): The device to create the noise tensors on.
        """
        self.size = size
        self.batch_size = batch_size
        self.device = device

        # Create a frequency grid for filtering
        h, w = self.size
        fy = torch.fft.fftfreq(h, device=self.device).view(-1, 1).repeat(1, w // 2 + 1)
        fx = torch.fft.fftfreq(w, device=self.device).view(1, -1).repeat(h, 1)
        # We only need up to the Nyquist frequency for the real FFT
        fx = fx[:, :w // 2 + 1]

        self.frequencies = torch.sqrt(fx**2 + fy**2)
        # Add a small epsilon to avoid division by zero at the DC component
        self.frequencies[0, 0] = 1e-8

    def _generate_colored_noise(self, exponent: float) -> torch.Tensor:
        """
        Generates noise with a power spectrum of 1/f^exponent.

        Args:
            exponent (float): The exponent for the power spectrum (e.g., 0=white, 1=pink, 2=brown).

        Returns:
            torch.Tensor: A batch of colored noise tensors.
        """
        # Generate white noise in the frequency domain
        # Use rfft for real-valued output
        noise_freq = torch.randn(self.batch_size, *self.size, device=self.device)
        noise_fft = torch.fft.rfft2(noise_freq)

        # Create the 1/f^exponent filter
        filter = self.frequencies**(-exponent / 2.0) # Apply to amplitude, so exponent/2

        # Apply the filter
        filtered_fft = noise_fft * filter.unsqueeze(0) # Add batch dimension to filter

        # Perform inverse FFT to get spatial noise
        spatial_noise = torch.fft.irfft2(filtered_fft, s=self.size)

        # Normalize the noise to have stddev=1 and mean=0
        std = torch.std(spatial_noise, dim=(-2, -1), keepdim=True)
        mean = torch.mean(spatial_noise, dim=(-2, -1), keepdim=True)
        normalized_noise = (spatial_noise - mean) / (std + 1e-8)

        return normalized_noise.unsqueeze(1) # Add channel dimension

    def generate_white_noise(self) -> torch.Tensor:
        """Generates standard white Gaussian noise."""
        return torch.randn(self.batch_size, 1, *self.size, device=self.device)

    def generate_pink_noise(self) -> torch.Tensor:
        """Generates pink noise (1/f spectrum)."""
        return self._generate_colored_noise(exponent=1.0)

    def generate_brown_noise(self) -> torch.Tensor:
        """Generates brown/red noise (1/f^2 spectrum)."""
        return self._generate_colored_noise(exponent=2.0)

    def generate_curriculum_noise(self, progress: float) -> torch.Tensor:
        """
        Generates noise with an exponent interpolated based on training progress.
        Starts with brown noise (exponent=2.0) and moves towards white noise (exponent=0.0).

        Args:
            progress (float): Training progress from 0.0 (start) to 1.0 (end).

        Returns:
            torch.Tensor: A batch of curriculum-shaped noise.
        """
        if not (0.0 <= progress <= 1.0):
            raise ValueError("Progress must be between 0.0 and 1.0")

        # Linearly interpolate the exponent from 2 (brown) to 0 (white)
        exponent = 2.0 * (1.0 - progress)
        return self._generate_colored_noise(exponent=exponent)