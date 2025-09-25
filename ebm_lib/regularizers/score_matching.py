import torch
from typing import Dict

from .base_regularizer import Regularizer
from ..ebm import EBM

class DenoisingScoreMatchingRegularizer(Regularizer):
    """
    Implements Denoising Score Matching (DSM) as a regularization loss.

    DSM provides an alternative way to train an EBM without MCMC. It works by
    perturbing the data with known noise and then training the EBM's score
    function (the gradient of the log-probability) to match the known score
    of the perturbed data distribution.

    When used as a regularizer, it can be added to the main CD objective to
    help shape the energy landscape around the data points.

    The loss is: (1/2) * || s_theta(x_tilde) - s_data(x_tilde) ||^2
    where s_theta is the model's score (-grad E(x)) and s_data is the
    analytically known score of the noised data distribution.
    """
    def __init__(self, config: Dict):
        """
        Initializes the DenoisingScoreMatchingRegularizer.

        Args:
            config (Dict): A dictionary containing configuration parameters.
                - lambda_dsm (float): The weight of this regularization loss.
                - sigma (float): The standard deviation of the Gaussian noise
                                 to add to the data.
        """
        super().__init__(config)
        self.lambda_dsm = config.get('lambda_dsm', 1.0)
        self.sigma = config.get('sigma', 0.1)

    def _get_noise(self, samples: torch.Tensor) -> torch.Tensor:
        """Helper method to generate noise, allows for easy mocking in tests."""
        return torch.randn_like(samples) * self.sigma

    def calculate_loss(self, ebm_model: EBM, positive_samples: torch.Tensor, negative_samples: torch.Tensor) -> torch.Tensor:
        """
        Calculates the DSM loss on the batch of positive (real) samples.

        Args:
            ebm_model (EBM): The EBM model.
            positive_samples (torch.Tensor): Samples from the data distribution.
            negative_samples (torch.Tensor): Not used by this regularizer.

        Returns:
            torch.Tensor: The scalar DSM loss.
        """
        # 1. Perturb the positive samples with Gaussian noise
        noise = self._get_noise(positive_samples)
        perturbed_samples = positive_samples + noise
        perturbed_samples.requires_grad_(True)

        # 2. Calculate the model's score (s_theta = -grad_x E(x))
        energy = ebm_model(perturbed_samples)
        model_score, = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=perturbed_samples,
            create_graph=True
        )
        model_score = -model_score

        # 3. Calculate the true score of the perturbed data distribution
        # s_data = - (x_tilde - x) / sigma^2
        true_score = -noise / (self.sigma**2)

        # 4. Calculate the score matching loss
        # The loss is the squared L2 distance between the scores, averaged over the batch.
        # We can sum the squared errors over features and then take the mean over the batch.
        loss = 0.5 * ((model_score - true_score.detach())**2).view(model_score.shape[0], -1).sum(dim=1).mean()

        return self.lambda_dsm * loss

    @staticmethod
    def get_hyperparameter_info() -> dict:
        return {
            "lambda_dsm": {
                "description": "Weight of the Denoising Score Matching loss.",
                "recommended": "1.0 is a common starting point.",
                "type": "float"
            },
            "sigma": {
                "description": "Standard deviation of the Gaussian noise added to the data.",
                "recommended": "0.01 to 0.1",
                "type": "float"
            }
        }