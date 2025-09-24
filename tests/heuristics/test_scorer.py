import pytest
import torch

from ebm_lib.heuristics.experts import HeuristicExpert
from ebm_lib.heuristics.scorer import MixtureOfExpertsScorer

class MockExpert(HeuristicExpert):
    """A mock expert that returns a predictable, non-normalized score."""
    def __init__(self, scores: torch.Tensor):
        super().__init__()
        self.scores = scores

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        # Returns a predefined set of scores for the batch
        if self.scores.shape[0] != image_batch.shape[0]:
            raise ValueError("MockExpert scores must match batch size")
        return self.scores

def test_scorer_init():
    """Tests the initialization of the MixtureOfExpertsScorer."""
    expert1 = MockExpert(torch.tensor([1.0]))
    expert2 = MockExpert(torch.tensor([1.0]))

    # Test with equal weights
    scorer = MixtureOfExpertsScorer(experts=[expert1, expert2])
    assert len(scorer.experts) == 2
    assert torch.allclose(scorer.weights, torch.tensor([1.0, 1.0]))

    # Test with custom weights
    scorer_weighted = MixtureOfExpertsScorer(experts=[expert1, expert2], weights=[0.2, 0.8])
    assert torch.allclose(scorer_weighted.weights, torch.tensor([0.2, 0.8]))

def test_scorer_raises_error_on_mismatched_weights():
    """Tests that an error is raised if the number of experts and weights don't match."""
    with pytest.raises(ValueError):
        MixtureOfExpertsScorer(experts=[MockExpert(torch.tensor([1.0]))], weights=[0.5, 0.5])

def test_scorer_forward_pass_and_normalization():
    """
    Tests the forward pass, ensuring that scores are normalized correctly
    before being combined.
    """
    # Expert 1 returns scores [10, 20, 30, 40]
    # Expert 2 returns scores [0.1, 0.2, 0.3, 0.4]
    # These are perfectly correlated, so their z-scores will be identical.
    expert1 = MockExpert(torch.tensor([10., 20., 30., 40.]))
    expert2 = MockExpert(torch.tensor([0.1, 0.2, 0.3, 0.4]))
    scorer = MixtureOfExpertsScorer(experts=[expert1, expert2])

    image_batch = torch.randn(4, 1, 8, 8) # Dummy batch of 4 images

    combined_scores = scorer(image_batch)

    assert combined_scores.shape == (4,)

    # The key property of z-score normalization is that the mean of the
    # combined scores should be very close to zero.
    assert torch.isclose(combined_scores.mean(), torch.tensor(0.0), atol=1e-6)

def test_scorer_weighting():
    """Tests that the weighting of experts works as expected."""
    # Expert 1 returns an increasing score
    # Expert 2 returns a decreasing score (uncorrelated with expert 1)
    scores1 = torch.tensor([10., 20., 30., 40.])
    scores2 = torch.tensor([40., 30., 20., 10.])
    expert1 = MockExpert(scores1)
    expert2 = MockExpert(scores2)

    image_batch = torch.randn(4, 1, 8, 8)

    # Heavily weight the second expert
    scorer = MixtureOfExpertsScorer(experts=[expert1, expert2], weights=[0.01, 0.99])
    combined_scores = scorer(image_batch)

    # Get scores from each expert and normalize them manually to compare
    norm_scores1 = (scores1 - scores1.mean()) / (scores1.std() + 1e-8)
    norm_scores2 = (scores2 - scores2.mean()) / (scores2.std() + 1e-8)

    # Since expert 2 has almost all the weight, the combined score should be
    # very close to the normalized score of expert 2.
    assert torch.allclose(combined_scores, norm_scores2, atol=0.1)
    # And very different from the scores of expert 1.
    assert not torch.allclose(combined_scores, norm_scores1, atol=0.1)
