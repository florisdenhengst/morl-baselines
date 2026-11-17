"""Diagnostic figures for the multi-user preference sampler, on the d=3 simplex.

Why this script exists
----------------------
This is the evidence for changing the study's preference sampler, kept so the decision can be re-examined.

Until 2026-11-17 `sample_user_weights` drew followers as `w_i ~ Dir(kappa * anchor)`, the Dirichlet written
by its *mean*. Because the concentration vector sums to exactly `kappa`, every component is below 1 unless
`kappa * min(anchor) > 1`, and a Dirichlet with any component below 1 has a density that **diverges on the
simplex boundary** rather than peaking at its own mean. The arithmetic mean was still exactly the anchor at
every `kappa`, so every summary statistic the study recorded looked sensible while the distribution was doing
something qualitatively different from the intended "stakeholders clustered around a shared direction".

`P(kappa * min(anchor) > 1)` over uniform anchors is 0.008 at d=4 and 0.000 at d>=6 for the main setting
`kappa=5`, and reaches only 0.292 at d=8 even at `kappa=50`. So the sweep was boundary-seeking at the
settings it actually used, and because `min(anchor)` shrinks with d, a fixed kappa placed different d in
different regimes, confounding the scaling figure's d axis.

The fix is to write the same family by its *mode* instead: `w_i ~ Dir(1 + kappa * c)`. For Dir(a) with every
a_j > 1 the mode is (a_j - 1) / (sum a - d), so the mode is exactly `c` at every kappa, and every component
is at least 1 by construction, whatever the centre. Nothing needs to be constrained or calibrated.

The two figures
---------------
Both plot the same quantity -- the analytic Dirichlet density over the 2-simplex, four decades per panel,
with 4000 draws scattered over it -- so they can be read against each other directly.

`preference_density.png`         the old family, `Dir(kappa * anchor)`.
`preference_density_mode.png`    the current family, `Dir(1 + kappa * c)`, at the same kappa and centres.

The swept kappa = 0 column shows what the mean parameterization could not do at all. `Dir(0)` is not a
distribution -- numpy silently returns an all-zero vector that is not a weight vector -- and the limit it
approaches is the *opposite* of uniform: at `kappa = 0.01`, 97% of draws put more than 0.99 weight on a
single objective. So the old family had no uniform setting to sweep to, whereas `1 + kappa * c` at kappa = 0
is exactly `Dir(1, ..., 1)`, a uniform draw over the simplex.

Rows are centre positions, from centred to near a vertex; columns are the swept `kappa in {1, 5, 50}`. Read
the left column of the first figure: mass piles onto the three edges and the centre sits in a hole of its own
distribution. The second figure peaks at the centre in every panel.

d=3 throughout, because that is the largest d that plots honestly without projection, and it already spans
both regimes of the old family: `kappa * min = kappa/3` at a centred anchor, so a centred panel is unimodal
at the swept kappa=5 while an off-centre one at the same kappa is not.
"""

import argparse
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import matplotlib
import numpy as np
from scipy.special import gammaln


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before pyplot is imported)

from preferences import expected_pairwise_distance  # noqa: E402


# Centre positions, chosen to walk from the best case to the worst along the quantity that decided the old
# family's regime, kappa * min(centre): 1/3, 0.20, 0.02 and 0.03 of kappa respectively.
CENTRES: List[Tuple[str, np.ndarray]] = [
    ("centred", np.array([1 / 3, 1 / 3, 1 / 3])),
    ("slightly off-centre", np.array([0.50, 0.30, 0.20])),
    ("near an edge", np.array([0.49, 0.49, 0.02])),
    ("near a vertex", np.array([0.90, 0.07, 0.03])),
]

# Simplex vertices in the plotting plane: objective 1 bottom left, 2 bottom right, 3 apex.
_VERTICES = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])

# Display dynamic range, in decades below each panel's own near-maximum. The mean-parameterized density
# genuinely goes to infinity on the boundary when alpha < 1, so a panel cannot be shown on an absolute
# scale; four decades keeps the mode and the boundary behaviour both legible.
_DECADES = 4.0


def _to_plane(weights: np.ndarray) -> np.ndarray:
    """Maps barycentric rows (n, 3) onto the plotting plane (n, 2)."""
    return weights @ _VERTICES


def _plane_grid(resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds a plane meshgrid and its barycentric coordinates, masked outside the triangle."""
    x, y = np.meshgrid(np.linspace(0.0, 1.0, resolution), np.linspace(0.0, np.sqrt(3) / 2, resolution))
    third = 2.0 * y / np.sqrt(3)
    second = x - y / np.sqrt(3)
    first = 1.0 - second - third
    barycentric = np.stack([first, second, third], axis=-1)
    # A margin rather than >= 0: the Dirichlet log density is infinite exactly on the boundary.
    barycentric[np.any(barycentric < 1e-4, axis=-1)] = np.nan
    return x, y, barycentric


def _dirichlet_log_density(barycentric: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Log Dirichlet density on the grid, up to the constant Jacobian of the plane embedding."""
    with np.errstate(invalid="ignore", divide="ignore"):
        log_normalizer = np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
        return np.sum((alpha - 1.0) * np.log(barycentric), axis=-1) - log_normalizer


def _boundary_fraction(samples: np.ndarray, floor: float = 0.01) -> float:
    """Fraction of drawn users that put less than `floor` weight on some objective.

    The comparable statistic across the two parameterizations. "Boundary-seeking" is otherwise easy to
    assert and hard to check: this is what it costs in stakeholders who effectively ignore an objective.
    """
    return float(np.mean(samples.min(axis=1) < floor))


def _draw_frame(axis, title: str, row_label: str = "") -> None:
    """Draws the triangle outline, objective labels and title; removes the cartesian axes.

    The per-panel annotation lives in the title rather than under the panel: these grids are tall, and text
    placed below a triangle reads as a caption for the row beneath it.
    """
    closed = np.vstack([_VERTICES, _VERTICES[:1]])
    axis.plot(closed[:, 0], closed[:, 1], color="0.25", linewidth=0.8)
    for position, label in zip(_VERTICES, ("$v_1$", "$v_2$", "$v_3$")):
        offset = (position - _VERTICES.mean(axis=0)) * 0.12
        axis.text(*(position + offset), label, ha="center", va="center", fontsize=7, color="0.3")
    axis.set_title(title, fontsize=7.5, linespacing=1.35)
    if row_label:
        axis.text(-0.06, 0.45, row_label, transform=axis.transAxes, rotation=90,
                  ha="center", va="center", fontsize=8.5, color="0.2")
    axis.set_aspect("equal")
    axis.set_xlim(-0.12, 1.12)
    axis.set_ylim(-0.1, np.sqrt(3) / 2 + 0.1)
    axis.axis("off")


def _show_field(axis, x: np.ndarray, y: np.ndarray, log_density: np.ndarray) -> None:
    """Renders a log-density field with a per-panel floor, so each panel shows its own shape."""
    finite = log_density[np.isfinite(log_density)]
    if finite.size == 0:
        return
    top = float(np.percentile(finite, 99.5))
    axis.contourf(
        x, y, np.clip(log_density, top - _DECADES, top),
        levels=np.linspace(top - _DECADES, top, 24), cmap="magma", extend="neither",
    )


def _write_grid(
    out: Path, name: str, suptitle: str, alpha_of: Callable[[np.ndarray, float], np.ndarray],
    annotate: Callable[[np.ndarray, float, float], str],
    kappas: Sequence[float], samples: int, resolution: int, seed: int,
) -> Path:
    """Renders one centre-by-kappa grid of simplex densities.

    Both figures differ only in how the concentration vector is built from (centre, kappa), so they share
    everything else: identical estimator, identical scaling, identical layout. That is what makes the two
    readable against one another.
    """
    figure, axes = plt.subplots(
        len(CENTRES), len(kappas), figsize=(2.75 * len(kappas), 3.15 * len(CENTRES)), squeeze=False
    )
    x, y, barycentric = _plane_grid(resolution)

    for row, (centre_name, centre) in enumerate(CENTRES):
        for column, kappa in enumerate(kappas):
            axis = axes[row][column]
            alpha = alpha_of(centre, kappa)
            heading = f"$\\kappa={kappa:g}$\n" if row == 0 else ""
            if alpha.min() <= 0.0:
                # Dir(0) is not a distribution. numpy does not raise for it -- it returns all-zero
                # "weights" that do not sum to 1 -- so the panel is named rather than drawn.
                _draw_frame(axis, heading + "$\\mathrm{Dir}(\\mathbf{0})$ is undefined\n"
                                            "no uniform panel exists\nin this parameterization",
                            row_label=centre_name if column == 0 else "")
                axis.text(0.5, 0.42, "--", transform=axis.transAxes, ha="center", va="center",
                          fontsize=16, color="0.6")
                continue
            rng = np.random.default_rng([seed, int(kappa * 1000), row])
            edge = _boundary_fraction(rng.dirichlet(alpha, size=100000))
            _draw_frame(axis, heading + annotate(alpha, kappa, edge),
                        row_label=centre_name if column == 0 else "")
            _show_field(axis, x, y, _dirichlet_log_density(barycentric, alpha))
            planar = _to_plane(rng.dirichlet(alpha, size=samples))
            axis.scatter(planar[:, 0], planar[:, 1], s=0.7, c="white", alpha=0.20, linewidths=0)
            centre_point = _to_plane(centre[None, :])[0]
            axis.scatter(*centre_point, marker="*", s=90, c="#38f9d7",
                         edgecolors="black", linewidths=0.4, zorder=5)

    figure.suptitle(suptitle, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.2)
    path = out / name
    figure.savefig(path, dpi=190)
    plt.close(figure)
    return path


def main() -> None:
    """Renders both diagnostic figures."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "figures", help="Output directory.")
    parser.add_argument("--kappas", type=float, nargs="+", default=[0.0, 5.0, 50.0], help="Swept kappas to show.")
    parser.add_argument("--samples", type=int, default=4000, help="Scatter draws per panel.")
    parser.add_argument("--resolution", type=int, default=320, help="Density grid resolution per side.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the draws.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    def annotate(alpha, kappa, edge):
        # The three regimes of a Dirichlet coordinate: alpha_j < 1 makes the density diverge as w_j -> 0,
        # alpha_j = 1 makes it flat in that coordinate, and alpha_j > 1 pushes it to zero at the boundary
        # and creates an interior mode. kappa = 0 is the all-ones case, i.e. exactly uniform.
        smallest = float(alpha.min())
        if smallest < 1.0:
            regime = "DIVERGES ON BOUNDARY"
        elif np.allclose(alpha, 1.0):
            regime = "uniform over the simplex"
        else:
            regime = "finite, peaks at centre"
        return (f"$\\min_j \\alpha_j={smallest:.2f}$ -- {regime}\n"
                f"RMS dist. $={expected_pairwise_distance(alpha):.2f}$\n"
                f"$P(\\min_j w_j<0.01)={edge:.3f}$")

    paths = [
        _write_grid(
            args.out, "preference_density.png",
            "Superseded: $w_i \\sim \\mathrm{Dir}(\\kappa\\, c)$, the Dirichlet by its mean\n"
            "star = centre; colour = analytic density, 4 decades per panel\n"
            "the density is unbounded on the boundary wherever $\\min_j \\kappa c_j < 1$",
            lambda centre, kappa: kappa * centre, annotate,
            args.kappas, args.samples, args.resolution, args.seed,
        ),
        _write_grid(
            args.out, "preference_density_mode.png",
            "Current: $w_i \\sim \\mathrm{Dir}(1 + \\kappa\\, c)$, the Dirichlet by its mode\n"
            "star = centre; colour = analytic density, 4 decades per panel\n"
            "every $\\alpha_j \\geq 1$ by construction, so the mode is exactly $c$ at every $\\kappa$",
            lambda centre, kappa: 1.0 + kappa * centre, annotate,
            args.kappas, args.samples, args.resolution, args.seed,
        ),
    ]
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
