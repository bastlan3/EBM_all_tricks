import torch
import torch.nn as nn
from typing import List, Optional

from .experts import HeuristicExpert

class MixtureOfExpertsScorer(nn.Module):
    """
    Combines scores from multiple heuristic experts into a single, unified score.

    This class takes a list of expert modules, runs an image batch through each,
    normalizes their scores to a common scale, and then computes a weighted
    average to produce a final score.
    """
    def __init__(self, experts: List[HeuristicExpert], weights: Optional[List[float]] = None):
        """
        Args:
            experts (List[HeuristicExpert]): A list of instantiated expert modules.
            weights (Optional[List[float]]): A list of weights for combining the expert
                                             scores. If None, all experts are weighted
                                             equally.
        """
        super().__init__()
        self.experts = nn.ModuleList(experts)

        if weights is not None:
            if len(experts) != len(weights):
                raise ValueError("The number of experts and weights must be the same.")
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            # Default to equal weighting if no weights are provided
            self.weights = torch.ones(len(experts), dtype=torch.float32)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Calculates a combined, normalized score for each image in a batch.

        The process is as follows:
        1. Get a raw score from each expert for every image in the batch.
        2. For each expert, normalize its scores across the batch (z-score normalization).
           This is crucial because different experts produce scores on different scales.
        3. Compute the weighted average of these normalized scores.

        Args:
            image_batch (torch.Tensor): A batch of images of shape (B, C, H, W).

        Returns:
            torch.Tensor: A tensor of combined scores of shape (B,).
        """
        if not self.experts:
            return torch.zeros(image_batch.shape[0], device=image_batch.device)

        # Ensure weights are on the same device as the input
        self.weights = self.weights.to(image_batch.device)

        # Collect raw scores from all experts
        all_scores = [expert(image_batch) for expert in self.experts]

        # Stack scores into a (num_experts, batch_size) tensor
        scores_tensor = torch.stack(all_scores, dim=0)

        # Normalize scores from each expert across the batch (z-score)
        # This brings all expert outputs to a comparable scale.
        mean = scores_tensor.mean(dim=1, keepdim=True)
        std = scores_tensor.std(dim=1, keepdim=True)
        # Add epsilon to avoid division by zero if all scores in a batch are the same
        normalized_scores = (scores_tensor - mean) / (std + 1e-8)

        # Reshape weights for broadcasting: (num_experts, 1)
        weights_reshaped = self.weights.view(-1, 1)

        # Compute the weighted average of the normalized scores
        # The denominator is the sum of weights, for a true weighted average.
        combined_score = (normalized_scores * weights_reshaped).sum(dim=0) / self.weights.sum()

        return combined_score
