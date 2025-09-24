from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

class HeuristicExpert(nn.Module, ABC):
    """
    Abstract base class for a heuristic expert that evaluates a property
    of an image, such as its noise level or sharpness.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Takes a batch of images and returns a scalar score for each image.

        Args:
            image_batch (torch.Tensor): A batch of images of shape (B, C, H, W).

        Returns:
            torch.Tensor: A tensor of scores of shape (B,).
        """
        pass

class LaplacianVarianceExpert(HeuristicExpert):
    """
    Estimates image sharpness by calculating the variance of the Laplacian.
    A higher variance generally corresponds to a sharper image with more defined edges.
    This can be used to distinguish between sharp and blurry/noisy images.
    """
    def __init__(self):
        super().__init__()
        # Define a fixed Laplacian kernel
        # This kernel is not learned; it's a fixed operator.
        kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        # Reshape for 2D convolution: (out_channels, in_channels, H, W)
        self.laplacian_kernel = kernel.view(1, 1, 3, 3)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Calculates the variance of the Laplacian for each image in the batch.
        """
        # Ensure kernel is on the same device as the input
        self.laplacian_kernel = self.laplacian_kernel.to(image_batch.device)

        # Assume input is (B, C, H, W). We process each channel independently.
        b, c, h, w = image_batch.shape

        # Reshape to apply convolution across all channels as if they were batches
        image_flat = image_batch.view(b * c, 1, h, w)

        # Apply Laplacian filter
        laplacian_filtered = F.conv2d(image_flat, self.laplacian_kernel, padding=1)

        # Calculate variance for each feature map
        variances = torch.var(laplacian_filtered, dim=(-1, -2))

        # Reshape back to (B, C) and average over channels
        variances_per_channel = variances.view(b, c)
        avg_variance = variances_per_channel.mean(dim=1)

        return avg_variance

class HighFrequencyEnergyExpert(HeuristicExpert):
    """
    Estimates image sharpness/noise by analyzing its Fourier spectrum.
    It calculates the proportion of energy contained in the high-frequency
    components of the image. More noise or detail corresponds to more
    high-frequency energy.
    """
    def __init__(self, high_freq_threshold: float = 0.25):
        """
        Args:
            high_freq_threshold (float): The fraction of the spectrum size to consider
                                         as the low-frequency region. E.g., 0.25 means
                                         the central 25% of frequencies are "low".
        """
        super().__init__()
        self.threshold = high_freq_threshold

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Calculates the ratio of high-frequency to total energy for each image.
        """
        # We only need one channel for this, so we convert to grayscale if needed.
        if image_batch.shape[1] > 1:
            # Simple average to convert to grayscale
            image_gray = image_batch.mean(dim=1, keepdim=True)
        else:
            image_gray = image_batch

        b, _, h, w = image_gray.shape

        # Compute 2D FFT
        fft_result = torch.fft.fft2(image_gray, dim=(-2, -1))
        # Shift zero frequency to the center
        fft_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))

        # Calculate magnitude spectrum (power)
        magnitude_spectrum = torch.abs(fft_shifted)**2

        # Create a mask for the low-frequency components
        mask = torch.ones_like(magnitude_spectrum)
        cy, cx = h // 2, w // 2
        low_freq_h = int(h * self.threshold) // 2
        low_freq_w = int(w * self.threshold) // 2
        mask[:, :, cy - low_freq_h : cy + low_freq_h, cx - low_freq_w : cx + low_freq_w] = 0

        # Calculate total energy and high-frequency energy
        total_energy = torch.sum(magnitude_spectrum, dim=(-1, -2))
        high_freq_energy = torch.sum(magnitude_spectrum * mask, dim=(-1, -2))

        # Avoid division by zero
        ratio = high_freq_energy / (total_energy + 1e-8)

        return ratio.squeeze(1) # Return shape (B,)
