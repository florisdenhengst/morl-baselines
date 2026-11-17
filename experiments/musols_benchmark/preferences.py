"""Randomized generation of multi-user preference matrices W, with a heterogeneity knob.

Why randomized preferences
---------------------------
Hand-picked stakeholder weights (as in `environments.py`) are useful for interpretable, domain-motivated case
studies, but they make a single arbitrary choice of how much the stakeholders (dis)agree. Since MUSOLS's
advantage depends directly on the size of the induced consensus polytope Omega_W = {W alpha | alpha in
Delta^{m-1}} -- unanimous stakeholders collapse it to a point, wildly divergent ones grow it towards the full
objective simplex -- any serious evaluation has to *sweep* stakeholder heterogeneity rather than fix it.

Sampling scheme
----------------
For m users over d objectives, given a concentration kappa >= 0:

    c   ~ Dirichlet(1, ..., 1)                     the latent panel centre
    w_i ~ Dirichlet(1 + kappa * c),  i = 1..m      the stakeholders, i.i.d. given c

The Dirichlet is parameterized by its **mode** rather than its mean. For Dir(a) with every a_j > 1 the mode is
(a_j - 1) / (sum_j a_j - d), so a = 1 + kappa * c puts the mode at exactly `c` for every kappa, and the users
are exchangeable -- no user is the panel's anchor.

  - kappa = 0  -> Dir(1, ..., 1), i.e. uniform over the objective simplex: maximal disagreement.
  - kappa -> oo -> the panel collapses onto `c`: a unanimous panel.

The swept levels are kappa in {0, 5, 50}, with 5 the main setting.

Why 1 + kappa*c rather than kappa*c
------------------------------------
The mean parameterization `w_i ~ Dir(kappa * c)` is the more obvious choice and was used until 2026-11-17. Its
concentration parameters sum to exactly kappa, so every component is below 1 unless `kappa * min(c) > 1`, and
a Dirichlet with any component below 1 has a density that **diverges on the simplex boundary** rather than
peaking anywhere near `c`: stakeholders get pushed towards ignoring objectives outright, and `c` sits at a
local density *minimum*. The arithmetic mean is still exactly `c`, which is why every statistic reported
looked sensible while the distribution did something else entirely.

That failure was not rare, and it bit hardest at the settings actually swept. `P(kappa * min(c) > 1)` over
uniform centres is 0.008 at d=4, kappa=5 and 0.000 at d>=6, kappa=5 -- so at the main setting the old family
was essentially never unimodal once there were more than four objectives -- and even at kappa=50 it reaches
only 0.292 at d=8. Because min(c) shrinks with d, a fixed kappa put different d in qualitatively different
regimes.

The mean parameterization also has no kappa=0: `Dir(0)` is not a distribution, and numpy returns an all-zero
vector for it rather than raising. Its kappa -> 0 limit is the opposite of uniform, collapsing onto a random
vertex -- at kappa=0.01, 97% of draws put over 0.99 weight on a single objective -- so the uniform panel that
anchors the heterogeneous end of the sweep was not reachable in that family at all.

Adding the 1 removes the failure mode by construction: every a_j = 1 + kappa * c_j is at least 1 whatever the
centre and whatever kappa, so the density is always unimodal at `c`. `plot_preference_sampler.py` renders both
parameterizations side by side. Measured fraction of users placing under 0.01 weight on some objective, d=3:

    centre position    kappa     Dir(kappa c)    Dir(1 + kappa c)
    centred                5            0.008               0.000
    near an edge           5            0.767               0.041
    near an edge          50            0.388               0.012

The cost, which should be stated when reporting: the panel's *arithmetic* mean is no longer exactly `c` (it is
(1 + kappa c) / (d + kappa), which blends towards uniform as kappa falls). This trades an exact mean for an
exact mode, and the mode is what a reader takes a density to be saying.

What kappa spans, and what it does not
---------------------------------------
Realized heterogeneity at each swept kappa, over random centres (m=2, 4000 panels per cell). RMS is the root
mean square pairwise L2 distance between two stakeholders' weight vectors, the quantity that sizes Omega_W:

    kappa:        0              5             50
    d=2:    0.572 / 0.781  0.324 / 0.923  0.117 / 0.990     (RMS / mean cosine)
    d=3:    0.567 / 0.705  0.371 / 0.862  0.136 / 0.981
    d=4:    0.543 / 0.660  0.373 / 0.820  0.152 / 0.971
    d=6:    0.491 / 0.605  0.368 / 0.752  0.160 / 0.954
    d=8:    0.441 / 0.585  0.350 / 0.709  0.165 / 0.937

Note how little the RMS spread moves across d at fixed kappa: 0.32 to 0.35 over d = 2..8 at the main setting
kappa=5. Under the mean parameterization the same column ran 0.34 to 0.51, growing by half as d grew, so this
change makes kappa a far more comparable knob across environments as a side effect of fixing the shape.

What it does not fix is the residual drift in *cosine* at fixed kappa. That is largely a property of cosine
rather than of the sampler: weight vectors lie on the simplex, so their norms shrink with d and a fixed L2
separation subtends a wider angle. Report realized agreement (`preference_statistics`) alongside nominal
kappa; `make_paper_assets.write_heterogeneity_figure` plots the realized value for exactly this reason.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class PreferencePanel:
    """A sampled stakeholder panel, with the latent centre that produced it.

    `centre` is not recoverable from `weights`: it is the mode of the distribution the columns were drawn
    from, and no user sits on it. It is carried out so the experiment record states it.

    Attributes:
        weights: W with shape (num_objectives, num_users); each column is one stakeholder's weight vector.
        centre: The latent panel centre c, shape (num_objectives,), kept at float64 (see `sample_user_weights`).
        concentration: The kappa the panel was drawn at.
    """

    weights: np.ndarray
    centre: np.ndarray
    concentration: float


def expected_pairwise_distance(alpha: np.ndarray) -> float:
    """RMS pairwise L2 distance between two independent Dir(alpha) draws, in closed form.

    For w ~ Dir(alpha) with a0 = sum_j alpha_j, Var(w_j) = alpha_j (a0 - alpha_j) / (a0^2 (a0 + 1)), and two
    independent draws give E||w - w'||^2 = 2 sum_j Var(w_j) = 2 (1 - sum_j alpha_j^2 / a0^2) / (a0 + 1).
    Verified against Monte Carlo to within 1.5% for both parameterizations over d = 2..8.

    Takes the raw concentration vector rather than (centre, kappa) so that it serves both the current
    `1 + kappa c` panels and the superseded `kappa c` ones, which `plot_preference_sampler.py` still draws
    for comparison.

    Args:
        alpha: Dirichlet concentration vector, shape (num_objectives,); strictly positive.

    Returns:
        float: RMS pairwise L2 distance.
    """
    total = float(np.sum(alpha))
    return float(np.sqrt(2.0 * (1.0 - np.sum(np.square(alpha)) / total**2) / (total + 1.0)))


def max_pairwise_distance(num_objectives: int) -> float:
    """The most heterogeneous panel this family can produce, reached at kappa = 0.

    kappa = 0 gives Dir(1, ..., 1), a uniform draw over the objective simplex, so the ceiling is
    sqrt(2 (1 - 1/d) / (d + 1)) -- 0.577, 0.577, 0.548, 0.488 and 0.441 for d = 2, 3, 4, 6, 8 -- and it does
    not depend on where the centre falls. Panels more dispersed than this would require stakeholders to repel
    each other, which is not a panel with a shared centre.

    Args:
        num_objectives: Number of objectives d.

    Returns:
        float: RMS pairwise L2 distance at kappa = 0.
    """
    return float(np.sqrt(2.0 * (1.0 - 1.0 / num_objectives) / (num_objectives + 1.0)))


def sample_user_weights(
    num_objectives: int,
    num_users: int,
    concentration: float,
    rng: np.random.Generator,
) -> PreferencePanel:
    """Samples a stakeholder panel of `num_users` weight vectors over `num_objectives` objectives.

    Args:
        num_objectives: Number of objectives d.
        num_users: Number of stakeholders m (>= 1).
        concentration: kappa >= 0, controlling how tightly stakeholders cluster around the latent centre.
            Larger means more agreement; 0 is a uniform draw over the simplex.
        rng: Seeded generator; the sole source of randomness, for reproducibility. The centre is drawn first
            and the users second, so a given generator state reproduces the panel exactly.

    Returns:
        PreferencePanel: W with shape (num_objectives, num_users), with the latent centre it was drawn around.
    """
    assert num_users >= 1, "num_users must be at least 1."
    assert concentration >= 0, "concentration must be non-negative."

    centre = rng.dirichlet(np.ones(num_objectives))
    weights = rng.dirichlet(1.0 + concentration * centre, size=num_users).T.astype(np.float32)
    # The centre stays float64 while W is cast to float32 to match the rest of the pipeline: the centre is a
    # distribution parameter, and `1 + kappa * c` loses most of a small component's significant digits in
    # float32, so a stored float32 centre would not reproduce the alpha the panel was actually drawn from.
    return PreferencePanel(weights=weights, centre=centre, concentration=concentration)


def preference_statistics(user_weights: np.ndarray) -> Dict[str, float]:
    """Summarizes the realized heterogeneity of a sampled preference matrix, for logging.

    The nominal kappa only fixes the sampling distribution; a paper should report the spread that actually
    materialized in each draw. All statistics are over the m*(m-1)/2 distinct stakeholder pairs (a
    single-stakeholder panel has no pairs and yields NaNs).

    Args:
        user_weights: W with shape (num_objectives, num_users).

    Returns:
        dict: mean/min pairwise cosine similarity (1.0 = unanimous panel), the maximum pairwise L2 distance
            between stakeholder weight vectors (i.e. the diameter of Omega_W), the realized RMS pairwise
            distance, and the realized *mean squared* pairwise distance.

    Both distance statistics are reported because only one of them pools correctly. The sampler fixes
    E||w_i - w_j||^2, so pooling across seeds means averaging `mean_squared_pairwise_distance` and taking the
    square root at the end. Averaging the per-panel `rms_pairwise_distance` instead understates the spread, by
    Jensen's inequality and badly when there are few pairs to average over. The per-panel RMS is kept because
    it is the readable per-cell number; it is not the one to average.
    """
    _, num_users = user_weights.shape
    cosines, distances = [], []
    for i in range(num_users):
        for j in range(i + 1, num_users):
            a, b = user_weights[:, i], user_weights[:, j]
            cosines.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
            distances.append(float(np.linalg.norm(a - b)))
    if not cosines:
        return {
            "mean_pairwise_cosine": float("nan"),
            "min_pairwise_cosine": float("nan"),
            "omega_w_diameter": 0.0,
            "rms_pairwise_distance": float("nan"),
            "mean_squared_pairwise_distance": float("nan"),
        }
    return {
        "mean_pairwise_cosine": float(np.mean(cosines)),
        "min_pairwise_cosine": float(np.min(cosines)),
        "omega_w_diameter": float(np.max(distances)),
        "rms_pairwise_distance": float(np.sqrt(np.mean(np.square(distances)))),
        "mean_squared_pairwise_distance": float(np.mean(np.square(distances))),
    }
