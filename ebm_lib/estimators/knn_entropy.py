import torch
import torch.nn as nn

class KnnEntropyEstimator(nn.Module):
    """
    Estimates the entropy of a batch of samples using a k-Nearest Neighbors (k-NN)
    based method. This module implements a proxy for entropy maximization by
    calculating a loss that encourages samples to be far apart from each other.

    The loss is calculated as the negative mean of the log distances to the
    k-th nearest neighbor for each sample in the batch. Maximizing entropy is
    equivalent to minimizing this loss.
    """
    def __init__(self, k: int = 5):
        """
        Args:
            k (int): The number of neighbors to consider for the estimation.
        """
        super().__init__()
        if k < 1:
            raise ValueError("k must be at least 1.")
        self.k = k

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Calculates the entropy loss for a batch of samples.

        Args:
            samples (torch.Tensor): A batch of samples, expected to be of shape
                                    (N, D) where N is the batch size and D is the
                                    feature dimension.

        Returns:
            torch.Tensor: A scalar tensor representing the entropy loss.
        """
        if samples.dim() > 2:
            # Flatten the samples if they are not already in (N, D) format
            samples = samples.view(samples.shape[0], -1)

        n, d = samples.shape

        if n <= self.k:
            # Cannot compute k-th neighbor if there are not enough samples
            return torch.tensor(0.0, device=samples.device)

        # Compute pairwise distances between all samples
        # The result is a (N, N) matrix of distances.
        dist_matrix = torch.cdist(samples, samples, p=2)

        # For each sample, find the distance to its k-th nearest neighbor.
        # We use topk with k+1 because the closest point is always the sample itself (distance 0).
        # We want the k-th closest *other* point.
        # We look for the k+1 smallest values along dimension 1 (for each row).
        # largest=False makes it find the smallest values.
        kth_distances = torch.topk(dist_matrix, k=self.k + 1, dim=1, largest=False).values

        # The k-th neighbor distance is the last one in the returned list.
        dist_to_kth_neighbor = kth_distances[:, -1]

        # To avoid log(0), we add a small epsilon.
        # The loss is the negative mean of the log distances. Minimizing this
        # loss maximizes the distances, thus maximizing the entropy.
        entropy_loss = -torch.log(dist_to_kth_neighbor + 1e-8).mean()

        return entropy_loss