"""Utility-based quality metrics for a returned (restricted) coverage set, in the multi-user setting.

CCS cardinality alone is a poor quality measure: a smaller set is the *point* of MUSOLS, so "fewer policies"
is only good if the set still serves every consensus a panel could reach. The field-standard answer is
utility-based evaluation (Zintgraf et al., 2015, "Quality Assessment of MORL Algorithms: A Utility-Based
Approach"), which asks how well a returned set serves a *distribution over utility functions* rather than how
many points it contains.

The only adaptation needed here is the distribution the metrics integrate over. Standard MORL evaluates over
weights drawn from the full objective simplex Delta^{d-1}; in the multi-user setting the decision maker can
only ever end up with a consensus weight w = W alpha for some alpha in Delta^{m-1}. Evaluating over the full
simplex would therefore penalize a restricted coverage set for failing to serve consensuses that, by
assumption, cannot arise. These helpers draw the weight set from the reachable consensus polytope Omega_W
instead and then delegate to `morl_baselines.common.performance_indicators`, so the underlying estimators stay
exactly the established, citable ones.
"""

from typing import Dict, List, Optional

import numpy as np

from morl_baselines.common.performance_indicators import (
    cardinality,
    expected_utility,
    maximum_utility_loss,
)


def sample_consensus_weights(user_weights: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Draws consensus weights w = W @ alpha with alpha uniform over the consensus simplex Delta^{m-1}.

    Args:
        user_weights: W with shape (num_objectives, num_users).
        num_samples: Number of consensus weights to draw.
        rng: Seeded generator, for reproducibility.

    Returns:
        np.ndarray: shape (num_samples, num_objectives); every row lies in Omega_W.
    """
    alphas = rng.dirichlet(np.ones(user_weights.shape[1]), size=num_samples)
    return (alphas @ user_weights.T).astype(np.float32)


def evaluate_coverage_set(
    ccs: List[np.ndarray],
    user_weights: np.ndarray,
    reference_set: Optional[List[np.ndarray]] = None,
    num_weight_samples: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """Scores a returned coverage set over the reachable consensus weight distribution.

    Args:
        ccs: The coverage set returned by an algorithm (payoff vectors in objective space).
        user_weights: W with shape (num_objectives, num_users), defining Omega_W.
        reference_set: Best-known or ground-truth set to measure utility loss against. When omitted, the
            utility-loss entry is NaN (there is nothing to be worse than).
        num_weight_samples: Consensus weights drawn to estimate the expectation/maximum.
        seed: Seed for the consensus weight draw. Fixed across algorithms within a trial so that every
            algorithm is scored on an identical weight set.

    Returns:
        dict: cardinality, expected consensus utility (higher is better) and maximum consensus utility loss
            (lower is better, 0.0 means the set is optimal for every sampled consensus).
    """
    if len(ccs) == 0:
        return {
            "cardinality": 0.0,
            "expected_consensus_utility": float("nan"),
            "max_consensus_utility_loss": float("nan"),
        }

    weights_set = sample_consensus_weights(user_weights, num_weight_samples, np.random.default_rng(seed))
    metrics = {
        "cardinality": float(cardinality(ccs)),
        "expected_consensus_utility": float(expected_utility(ccs, weights_set)),
        "max_consensus_utility_loss": float("nan"),
    }
    if reference_set is not None and len(reference_set) > 0:
        metrics["max_consensus_utility_loss"] = float(maximum_utility_loss(ccs, reference_set, weights_set))
    return metrics


def exact_restricted_ccs(candidates: np.ndarray, user_weights: np.ndarray, num_weight_samples: int = 20000, seed: int = 0):
    """Brute-force ground-truth restricted CCS, for environments whose outcome set is fully enumerable.

    For the exact-solver environments (whose complete set of attainable payoff vectors is known), densely
    sampling Omega_W and recording which candidate is optimal for each consensus weight recovers the true
    restricted CCS, independent of any search algorithm. This is the reference set the utility-loss metric
    should be measured against wherever it is available.

    Args:
        candidates: All attainable payoff vectors, shape (num_candidates, num_objectives).
        user_weights: W with shape (num_objectives, num_users).
        num_weight_samples: Consensus weights to sample; denser is more faithful.
        seed: Seed for the consensus weight draw.

    Returns:
        List[np.ndarray]: the payoff vectors optimal for at least one sampled consensus weight.
    """
    weights_set = sample_consensus_weights(user_weights, num_weight_samples, np.random.default_rng(seed))
    best = np.unique(np.argmax(weights_set @ np.asarray(candidates).T, axis=1))
    return [np.asarray(candidates)[i] for i in best]
