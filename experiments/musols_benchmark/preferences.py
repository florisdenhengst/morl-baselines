"""Randomized generation of multi-user preference matrices W, with a controllable heterogeneity knob.

Why randomized preferences
---------------------------
Hand-picked stakeholder weights (as in `environments.py`) are useful for interpretable, domain-motivated case
studies, but they make a single arbitrary choice of how much the stakeholders (dis)agree. Since MUSOLS's
advantage depends directly on the size of the induced consensus polytope Omega_W = {W alpha | alpha in
Delta^{m-1}} -- unanimous stakeholders collapse it to a point, wildly divergent ones grow it towards the full
objective simplex -- any serious evaluation has to *sweep* stakeholder heterogeneity rather than fix it.

Sampling scheme
----------------
For m users over d objectives, given a concentration parameter kappa > 0:

    w_1 ~ Dirichlet(1, ..., 1)              (uniform over the objective simplex: the anchor stakeholder)
    w_i ~ Dirichlet(kappa * w_1),  i > 1    (the remaining stakeholders, centered on the anchor)

Dirichlet(kappa * w_1) has mean exactly w_1 (its concentration parameters sum to kappa, since w_1 sums to 1),
so kappa is a pure spread control that leaves the expected preference direction fixed:

  - large kappa  -> stakeholders cluster tightly around the anchor (near-unanimous panel; Omega_W is tiny)
  - small kappa  -> stakeholders scatter across the simplex (deeply divided panel; Omega_W approaches
                    the convex hull of near-vertex preferences)

This gives a single, interpretable axis to sweep in the experiments, and -- because the anchor itself is drawn
uniformly -- it does not privilege any particular region of objective space. Both draws come from one seeded
`numpy.random.Generator`, so a (seed, m, kappa) triple reproduces a preference matrix exactly.

Choosing a sweep range: the mapping from kappa to realized agreement depends on d, so the informative range
shifts with the environment. Measured mean pairwise cosine similarity (m=3, 300 draws per cell):

    kappa:      1      2      5     10     50    200   1000
    d=2:     0.866  0.903  0.955  0.973  0.994  0.999  1.000
    d=4:     0.705  0.777  0.864  0.919  0.980  0.995  0.999
    d=6:     0.584  0.691  0.790  0.865  0.966  0.991  0.998

So kappa in roughly [1, 50] spans "deeply divided" to "near-unanimous" for d>=4, while low-dimensional
problems need smaller kappa to produce genuine disagreement. Because any single draw is noisy -- especially in
low dimensions -- experiments should report the *realized* spread from `preference_statistics`, not just the
nominal kappa.
"""

from typing import Dict

import numpy as np


# Dirichlet concentration parameters must be strictly positive; the anchor's smallest components can be
# vanishingly small, so floor them rather than risk a degenerate (or erroring) draw.
_MIN_CONCENTRATION = 1e-6


def sample_user_weights(
    num_objectives: int,
    num_users: int,
    concentration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Samples a user preference matrix W of shape (num_objectives, num_users).

    Args:
        num_objectives: Number of objectives d.
        num_users: Number of stakeholders m (>= 1).
        concentration: Dirichlet concentration kappa > 0 controlling how tightly the non-anchor stakeholders
            cluster around the uniformly-drawn anchor. Larger means more agreement.
        rng: Seeded generator; the sole source of randomness, for reproducibility.

    Returns:
        np.ndarray: W with shape (num_objectives, num_users); each column is one stakeholder's weight vector,
            lies in the objective simplex, and sums to 1.
    """
    assert num_users >= 1, "num_users must be at least 1."
    assert concentration > 0, "concentration must be positive."

    anchor = rng.dirichlet(np.ones(num_objectives))
    columns = [anchor]
    if num_users > 1:
        alpha = np.maximum(concentration * anchor, _MIN_CONCENTRATION)
        columns.extend(rng.dirichlet(alpha) for _ in range(num_users - 1))
    return np.stack(columns, axis=1).astype(np.float32)


def preference_statistics(user_weights: np.ndarray) -> Dict[str, float]:
    """Summarizes the realized heterogeneity of a sampled preference matrix, for logging.

    The nominal concentration kappa only fixes the sampling distribution; a paper should report the spread
    that actually materialized in each draw. All statistics are over the m*(m-1)/2 distinct stakeholder pairs
    (a single-stakeholder panel has no pairs and yields NaNs).

    Args:
        user_weights: W with shape (num_objectives, num_users).

    Returns:
        dict: mean/min pairwise cosine similarity (1.0 = unanimous panel) and the maximum pairwise L2 distance
            between stakeholder weight vectors, i.e. the diameter of Omega_W.
    """
    _, num_users = user_weights.shape
    cosines, distances = [], []
    for i in range(num_users):
        for j in range(i + 1, num_users):
            a, b = user_weights[:, i], user_weights[:, j]
            cosines.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
            distances.append(float(np.linalg.norm(a - b)))
    if not cosines:
        return {"mean_pairwise_cosine": float("nan"), "min_pairwise_cosine": float("nan"), "omega_w_diameter": 0.0}
    return {
        "mean_pairwise_cosine": float(np.mean(cosines)),
        "min_pairwise_cosine": float(np.min(cosines)),
        "omega_w_diameter": float(np.max(distances)),
    }
