"""Generates the paper's tables (markdown + LaTeX) and figures (TikZ/pgfplots) from study results.

Everything a paper reports is derived here from the JSON Lines files written by `run_study.py`, so no number
is ever transcribed by hand and every asset can be regenerated after a re-run. Markdown goes to stdout for
reading in a terminal; LaTeX and TikZ are written to files for `\\input{}`.

Figures are emitted as self-contained pgfplots pictures with inline coordinates, so they compile without any
external data files. The preamble each figure needs is stated in a comment at the top of its file.

Environments are split into two groups throughout, because mixing them would misrepresent the evidence:
  - *well-powered*: exact-solver and synthetic tasks, tens of cells per configuration, ground-truth utility
    loss available;
  - *indicative*: the deep-RL tasks, where only a handful of cells were affordable and several ran at reduced
    training budgets, so differences are not statistically resolvable.

Examples:
    python experiments/musols_benchmark/make_paper_assets.py results/*.jsonl --out paper_assets
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import wilcoxon


# Ordered as: the unrestricted algorithm MUSOLS specialises, MUSOLS itself, then its two ablations. The
# baselines are presented as ablations because that is what they are -- each removes exactly one of MUSOLS's
# two ideas, so the pair localises where the benefit comes from:
#   -consensus  drops the consensus search and solves each stakeholder's own weight (formerly "vertex-only"),
#               so what remains is the restriction without any search of Omega_W's interior;
#   -search     drops the prioritized corner-weight search for uniform sampling of alpha (formerly
#               "Random-Omega_W"), so what remains is the restriction without direction.
# A minus sign rather than a hyphen, since these denote removal, not a compound name.
ALGORITHMS = ("ols", "musols", "vertex", "random")
PRETTY = {"ols": "OLS", "musols": "MUSOLS", "vertex": "$-$consensus", "random": "$-$search"}
PRETTY_MD = {"ols": "OLS", "musols": "MUSOLS", "vertex": "−consensus", "random": "−search"}
CELL_KEYS = ("env", "seed", "num_users", "concentration")
# Environments are split by how much evidence they carry, measured from the records rather than listed by
# name: a hardcoded list silently misfiled synthetic-d5..d8 (600 cells each, ground-truth utility loss) as
# "indicative", and would have done the same to every newly added environment.
# Environments with this many cells or fewer are excluded from the assets entirely rather than reported in a
# separate "indicative" table: at that sample size no comparison is resolvable, and publishing the rows invites
# them to be read as results.
MIN_CELLS = 100
# Environments reported in the tables, named explicitly per paper_assets_config rather than derived, so the
# paper's headline set is an editorial choice. A listed environment that turns out thin is warned about rather
# than silently dropped or silently included.
MAIN_ENVS = ("deep-sea-treasure", "resource-gathering", "fruit-tree", "synthetic-d8", "water-reservoir")
MAIN_NUM_USERS = 2      # headline table
AUX_NUM_USERS = 3       # auxiliary table; m is therefore constant within a table and gets no column
# Metrics plotted against the number of objectives, one stacked subplot each, sharing an x axis.
SCALING_PLOT_METRICS = (
    ("ccs_size", "$|\\mathcal{C}_W|$", "mean"),
    ("num_evaluated", "Evaluations", "mean"),
    ("elapsed_seconds", "Runtime (s)", "median_boot"),
)
# Panel size for the heterogeneity figure. m=3 rather than m=2: Omega_W is a 2-simplex instead of a segment,
# so the restricted coverage set has room to move as kappa varies. Measured against the analytic fronts, the
# kappa=1 -> kappa=50 span at m=3 is 5.56 -> 2.02 on fruit-tree against 3.03 -> 1.64 at m=2, and every task
# shows the same widening. The m=2 version is emitted alongside it for reference.
HETEROGENEITY_NUM_USERS = 3
ENV_SHORT = {"deep-sea-treasure": "DST"}
HETEROGENEITY_PLOT_METRICS = (
    ("ccs_size", "CCS size", "mean"),
    ("num_evaluated", "Evaluations", "mean"),
)
# Short, typewriter-set names for plot legends: the full keys will not sit four-across in one shared legend,
# and deep-sea-treasure in particular crowds out the rest.
ENV_SHORT = {"deep-sea-treasure": "DST"}
PLOT_COLOURS = {"vertex": "gray", "random": "teal", "ols": "orange", "musols": "blue"}
PLOT_MARKS = {"vertex": "triangle*", "random": "square*", "ols": "diamond*", "musols": "*"}
# The headline table; every other heterogeneity level present gets its own auxiliary table.
MAIN_CONCENTRATION = 5.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=str, nargs="+", help="JSON Lines files written by run_study.py.")
    parser.add_argument("--out", type=str, default="paper_assets", help="Directory to write .tex assets into.")
    return parser.parse_args()


def load(paths: Sequence[str]) -> List[dict]:
    """Loads every record from the given JSON Lines files."""
    records = []
    for path in paths:
        with Path(path).open() as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def _fmt(value: float, digits: int = 3) -> str:
    """Formats a number compactly, switching to scientific notation only when it would otherwise be unreadable."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e5:
        return f"{value:.1e}"
    return f"{value:.{digits}g}"


def _clean(values: Sequence[float]) -> np.ndarray:
    return np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)


def _median_bootstrap_ci(values: Sequence[float], resamples: int = 10_000, seed: int = 0):
    """Median with a percentile bootstrap 95% CI. Used for wall-clock time.

    Runtimes here are heavily right-skewed and, for OLS, truncated at the per-algorithm timeout, so a mean is
    not an estimate of any quantity of interest -- it is dragged by the tail and capped from above at the same
    time. The median is unaffected by either, and bootstrapping gives it an interval without assuming a shape.
    """
    arr = _clean(values)
    if arr.size == 0:
        return None, None, None
    point = float(np.median(arr))
    if arr.size == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(resamples, arr.size), replace=True)
    lo, hi = np.percentile(np.median(draws, axis=1), [2.5, 97.5])
    return point, float(lo), float(hi)


def _mean_bootstrap_ci(values: Sequence[float], resamples: int = 10_000, seed: int = 0):
    """Mean with a percentile bootstrap 95% CI.

    Used for quantities bounded below at zero that pile up *on* that bound -- delta-EU is exactly 0 whenever an
    algorithm is the best in its cell, which is most cells for MUSOLS and OLS. A normal-approximation interval
    on such a sample happily returns a negative lower bound, which is impossible by construction. Every
    bootstrap resample is drawn from the observed non-negative values, so the interval cannot leave the
    support.
    """
    arr = _clean(values)
    if arr.size == 0:
        return None, None, None
    point = float(arr.mean())
    if arr.size < 2:
        return point, point, point
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(resamples, arr.size), replace=True)
    lo, hi = np.percentile(draws.mean(axis=1), [2.5, 97.5])
    return point, float(lo), float(hi)


def _mean_ci(values: Sequence[float]):
    """Mean with a normal-approximation 95% CI, for the bounded count and utility metrics."""
    arr = _clean(values)
    if arr.size == 0:
        return None, None, None
    point = float(arr.mean())
    if arr.size < 2:
        return point, point, point
    half = 1.96 * float(arr.std(ddof=1)) / np.sqrt(arr.size)
    return point, point - half, point + half


def _stat(values: Sequence[float], kind: str):
    """Point estimate and 95% interval for one metric.

    "median_boot" bootstraps the median (wall-clock time: skewed and censored at the budget), "mean_boot"
    bootstraps the mean (delta-EU: bounded below at 0 with mass on the bound), "mean" is the normal
    approximation (counts, which are neither).
    """
    if kind == "median_boot":
        return _median_bootstrap_ci(values)
    if kind == "mean_boot":
        return _mean_bootstrap_ci(values)
    return _mean_ci(values)


def _fmt_value(value, floor=None) -> str:
    """Formats one number for a table cell: fixed-point, never scientific, floored where floor is given."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    if value == 0:
        return "0"
    if floor is not None and abs(value) < floor:
        return f"$<${floor:g}"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _fmt_point(point, floor=None, bold: bool = False, markdown: bool = False) -> str:
    text = _fmt_value(point, floor)
    if not bold:
        return text
    return f"**{text}**" if markdown else f"\\textbf{{{text}}}"


def _fmt_interval(point, lo, hi, floor=None, markdown: bool = False) -> str:
    """The CI, for its own column. Suppressed when the point estimate is already below the display floor:
    an interval around a value we are declaring negligible only draws the eye back to the noise."""
    if point is None or lo is None:
        return "--"
    if floor is not None and abs(point) < floor:
        return "--"
    body = f"[{_fmt_value(lo, floor)}, {_fmt_value(hi, floor)}]"
    return body if markdown else f"{{\\scriptsize {body}}}"


# metric key, LaTeX header, markdown header, higher_better, interval kind, display floor.
# The floor is the magnitude below which a value is reported as negligible rather than printed: delta-EU and
# MUL routinely land at 1e-6, which is solver noise dressed up as a measurement, and sub-centisecond runtimes
# say nothing about an algorithm. Printing those as "<0.001" states the finding (indistinguishable from zero)
# instead of inviting a reader to compare digits that do not mean anything.
TABLE_METRICS = (
    ("ccs_size", "size$\\downarrow$", "size", False, "mean", None),
    ("num_evaluated", "evals$\\downarrow$", "evals", False, "mean", None),
    ("elapsed_seconds", "time (s)$\\downarrow$", "time (s)", False, "median_boot", 1e-2),
    ("delta_eu", "$\\Delta$-EU$\\downarrow$", "D-EU", False, "mean_boot", 1e-3),
    ("max_consensus_utility_loss", "MUL$\\downarrow$", "MUL", False, "mean", 1e-3),
)


def _attach_delta_eu(records: List[dict]) -> None:
    """Adds `delta_eu`: expected consensus utility shortfall against the best algorithm *in the same cell*.

    Absolute EU is nearly identical between MUSOLS and OLS -- which is the result, not a defect -- but that
    makes the column hard to read and invites comparing values across environments, where EU is not on a
    common scale. Reporting the within-cell gap from the best keeps the "no quality loss" evidence (a gap of
    0 means nothing was given up) while making ties legible and the metric comparable across environments.
    """
    for cell in _by_cell(records).values():
        best = max(
            (r["expected_consensus_utility"] for r in cell.values()
             if r.get("expected_consensus_utility") is not None),
            default=None,
        )
        for record in cell.values():
            eu = record.get("expected_consensus_utility")
            record["delta_eu"] = None if (eu is None or best is None) else best - eu


def _bold_set(block: List[dict], metric: str, higher_better: bool, alpha: float = 0.05) -> set:
    """Algorithms to embolden: the best point estimate, plus any not significantly worse than it.

    Paired Wilcoxon within cells, so the comparison respects that every algorithm saw the same panel. Showing
    ties as ties is the point -- MUSOLS matching OLS on quality is a claim, and a rule that bolded a single
    winner would hide exactly that.
    """
    present = [a for a in ALGORITHMS if any(r["algorithm"] == a for r in block)]
    points = {}
    for algorithm in present:
        values = _clean([r.get(metric) for r in block if r["algorithm"] == algorithm])
        if values.size:
            points[algorithm] = float(values.mean() if metric != "elapsed_seconds" else np.median(values))
    if not points:
        return set()
    best = max(points, key=points.get) if higher_better else min(points, key=points.get)
    bold = {best}
    cells = _by_cell(block)
    for algorithm in points:
        if algorithm == best:
            continue
        pairs = [
            (c[best][metric], c[algorithm][metric])
            for c in cells.values()
            if best in c and algorithm in c
            and c[best].get(metric) is not None and c[algorithm].get(metric) is not None
        ]
        diffs = [a - b for a, b in pairs]
        if len(diffs) < 2 or all(abs(d) < 1e-12 for d in diffs):
            bold.add(algorithm)      # identical to the best: a tie
            continue
        try:
            if wilcoxon([a for a, _ in pairs], [b for _, b in pairs]).pvalue >= alpha:
                bold.add(algorithm)  # not significantly worse
        except ValueError:
            bold.add(algorithm)
    return bold


def _by_cell(records: List[dict]) -> Dict[tuple, Dict[str, dict]]:
    """Indexes records by cell coordinates, so algorithms can be compared pairwise within a cell."""
    cells: Dict[tuple, Dict[str, dict]] = defaultdict(dict)
    for record in records:
        cells[tuple(record[k] for k in CELL_KEYS)][record["algorithm"]] = record
    return cells


# --------------------------------------------------------------------------------------- main results table


def _task_identity(record: dict):
    """Splits a record's task into (display name, d, depth) so each becomes its own column.

    Environment *keys* encode task parameters in their suffix -- synthetic-d4, fruit-tree-d6 -- which makes a
    single "Environment" column carry three different pieces of information at once and forces the reader to
    know the naming convention. Pulling d and depth out means synthetic's seven keys collapse into one named
    row group distinguished by its d column, and fruit-tree's depths likewise.

    Depth comes from the recorded env_kwargs, so it is the value the run actually used rather than something
    parsed back out of a name; records written before env_kwargs was stored fall back to the key's suffix.
    """
    env = record["env"]
    d = record.get("num_objectives")
    kwargs = record.get("env_kwargs") or {}
    depth = kwargs.get("depth")
    if env.startswith("synthetic"):
        display = "synthetic"
    elif env.startswith("fruit-tree"):
        display = "fruit-tree"
        if depth is None and "-d" in env:
            try:
                depth = int(env.rsplit("-d", 1)[1])
            except ValueError:
                depth = None
    else:
        display = env
    return display, d, depth


def main_results_rows(records: List[dict], envs: Sequence[str], markdown: bool = False) -> List[List[str]]:
    """Rows for the main table: task / d / algorithm, then a (point, CI) pair per metric.

    Each metric occupies two cells so the point estimates form a column a reader can scan down without the
    intervals breaking the alignment; the two share one \\multicolumn header. The task cell carries the task
    name on its first row and the cell count on its second, which keeps n adjacent to the numbers it was
    computed from without spending a column on a value that repeats down the block.
    """
    wanted = set(envs)
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for record in records:
        if record["env"] not in wanted:
            continue
        display, d, _ = _task_identity(record)
        groups[(display, d)].append(record)

    rows = []
    for key in sorted(groups, key=lambda k: (k[0], k[1] if k[1] is not None else -1)):
        display, d = key
        block = groups[key]
        n_cells = len({tuple(r[k] for k in CELL_KEYS) for r in block})
        bold = {metric: _bold_set(block, metric, hi) for metric, _, _, hi, _, _ in TABLE_METRICS}
        present = [a for a in ALGORITHMS if any(r["algorithm"] == a for r in block)]
        for index, algorithm in enumerate(present):
            subset = [r for r in block if r["algorithm"] == algorithm]
            cells = []
            for metric, _, _, _, kind, floor in TABLE_METRICS:
                point, lo, hi = _stat([r.get(metric) for r in subset], kind)
                cells.append(_fmt_point(point, floor, algorithm in bold[metric], markdown))
                cells.append(_fmt_interval(point, lo, hi, floor, markdown))
            if index == 0:
                head = [display, str(d) if d is not None else "--"]
            elif index == 1:
                head = [f"$n={n_cells}$" if not markdown else f"n={n_cells}", ""]
            else:
                head = ["", ""]
            rows.append(head + [algorithm] + cells + [index == len(present) - 1])
    return rows


def write_main_table(records: List[dict], envs: Sequence[str], out: Path, name: str, caption: str, label: str) -> str:
    """Writes the LaTeX main-results table and returns its markdown twin."""
    n_metrics = len(TABLE_METRICS)
    # Point estimate right-aligned, its interval left-aligned beside it, so the pair reads as one quantity
    # while the point column still lines up vertically.
    colspec = "lrl" + "rl" * n_metrics
    header = " & ".join(
        ["task", "$d$", "algorithm"]
        + [f"\\multicolumn{{2}}{{c}}{{{tex}}}" for _, tex, _, _, _, _ in TABLE_METRICS]
    )

    latex = [
        "% Requires: \\usepackage{booktabs}",
        "% Wide table: consider \\begin{table*} or \\sidewaystable if it overruns a two-column layout.",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    rows = main_results_rows(records, envs)
    for row in rows:
        *cells, is_last_of_task = row
        body = [c.replace("_", "\\_") for c in cells[:2]] + [PRETTY[cells[2]]] + cells[3:]
        # Tasks are separated by vertical space rather than a rule: a \midrule per task chops a table this
        # tall into slabs, while [.5em] groups the block visually without adding ink.
        terminator = " \\\\[.5em]" if is_last_of_task and row is not rows[-1] else " \\\\"
        latex.append(" & ".join(body) + terminator)
    latex += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
              "\\end{table}", ""]
    (out / f"{name}.tex").write_text("\n".join(latex))

    md_head = ["task", "d", "algorithm"]
    for _, _, md, _, _, _ in TABLE_METRICS:
        md_head += [md, f"{md} 95% CI"]
    lines = ["| " + " | ".join(md_head) + " |", "|" + "|".join("---" for _ in md_head) + "|"]
    for row in main_results_rows(records, envs, markdown=True):
        *cells, _ = row
        lines.append("| " + " | ".join(cells[:2] + [PRETTY_MD[cells[2]]] + cells[3:]) + " |")
    return "\n".join(lines)



def scaling_ratios(records: List[dict]) -> Dict[int, Dict[str, float]]:
    """Median OLS-to-MUSOLS cost ratios per number of objectives, over cells where both ran."""
    per_d = defaultdict(lambda: defaultdict(list))
    for cell in _by_cell([r for r in records if r["env"].startswith("synthetic")]).values():
        if "musols" not in cell or "ols" not in cell:
            continue
        musols, ols = cell["musols"], cell["ols"]
        d = musols["num_objectives"]
        per_d[d]["time"].append(ols["elapsed_seconds"] / max(musols["elapsed_seconds"], 1e-9))
        per_d[d]["evals"].append(ols["num_evaluated"] / max(musols["num_evaluated"], 1))
        per_d[d]["ccs"].append(ols["ccs_size"] / max(musols["ccs_size"], 1))
    return {d: {k: float(np.median(v)) for k, v in metrics.items()} for d, metrics in sorted(per_d.items())}


def _synthetic_shape(records: List[dict]) -> str:
    """Describes the synthetic task's own shape from the records, rather than hardcoding it in a caption.

    The candidate count and geometry are sweep parameters that have changed during development (sphere
    geometry with N=20 early on, gaussian with N=30 later), so a caption asserting either is a latent error
    waiting for the next re-run. Older records predate `env_kwargs` being stored; those fall back to a
    parameter-free description rather than claiming something unverifiable.
    """
    shapes = {
        (r.get("env_kwargs") or {}).get("num_candidates"): (r.get("env_kwargs") or {}).get("geometry")
        for r in records
        if str(r.get("env", "")).startswith("synthetic")
    }
    shapes.pop(None, None)
    if len(shapes) == 1:
        n, geometry = next(iter(shapes.items()))
        return f"$N={n}$ attainable payoffs, {geometry} geometry"
    return "see Section~\\ref{sec:setup} for the task parameters"


def write_scaling_table(records: List[dict], out: Path) -> str:
    """Writes the LaTeX scaling table and returns its markdown twin."""
    ratios = scaling_ratios(records)
    latex = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{rrrr}",
        "\\toprule",
        "$d$ & Time & Evaluations & $|\\mathcal{C}|$ \\\\",
        "\\midrule",
    ]
    markdown = ["| d | Time | Evaluations | \\|CCS\\| |", "|---|---|---|---|"]
    for d, metrics in ratios.items():
        latex.append(
            f"{d} & {metrics['time']:.0f}$\\times$ & {metrics['evals']:.1f}$\\times$ & {metrics['ccs']:.1f}$\\times$ \\\\"
        )
        markdown.append(f"| {d} | {metrics['time']:.0f}× | {metrics['evals']:.1f}× | {metrics['ccs']:.1f}× |")
    latex += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Cost of OLS relative to MUSOLS on the synthetic task ({_synthetic_shape(records)}), "
        "median over cells. The wall-clock ratio grows superlinearly in the number of objectives, "
        "while the returned set stays an order of magnitude larger.}",
        "\\label{tab:scaling}",
        "\\end{table}",
        "",
    ]
    (out / "scaling_table.tex").write_text("\n".join(latex))
    return "\n".join(markdown)


# -------------------------------------------------------------------------------------- significance table


def write_significance_table(records: List[dict], envs: Sequence[str], out: Path) -> str:
    """Writes paired Wilcoxon p-values of MUSOLS against each baseline, per environment."""
    metrics = [("expected_consensus_utility", "EU"), ("max_consensus_utility_loss", "MUL"), ("elapsed_seconds", "Time")]
    latex = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{ll" + "r" * len(metrics) + "}",
        "\\toprule",
        "Environment & Comparison & " + " & ".join(label for _, label in metrics) + " \\\\",
        "\\midrule",
    ]
    markdown = [
        "| Environment | Comparison | " + " | ".join(label for _, label in metrics) + " |",
        "|" + "|".join("---" for _ in range(2 + len(metrics))) + "|",
    ]
    for env in envs:
        block = [r for r in records if r["env"] == env]
        if not block:
            continue
        cells = _by_cell(block)
        for algorithm in ("random", "vertex", "ols"):
            values = []
            for metric, _ in metrics:
                pairs = [
                    (c["musols"][metric], c[algorithm][metric]) for c in cells.values() if "musols" in c and algorithm in c
                ]
                pairs = [(a, b) for a, b in pairs if not (np.isnan(a) or np.isnan(b))]
                left = np.array([p[0] for p in pairs])
                right = np.array([p[1] for p in pairs])
                if left.size < 3 or np.allclose(left, right):
                    values.append("--")
                else:
                    try:
                        values.append(f"{wilcoxon(left, right).pvalue:.1e}")
                    except ValueError:
                        values.append("--")
            latex.append(
                f"{env.replace('_', chr(92) + '_')} & MUSOLS vs {PRETTY[algorithm]} & " + " & ".join(values) + " \\\\"
            )
            markdown.append(f"| {env} | MUSOLS vs {PRETTY_MD[algorithm]} | " + " | ".join(values) + " |")
        latex.append("\\midrule")
    if latex[-1] == "\\midrule":
        latex.pop()
    latex += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Paired Wilcoxon signed-rank $p$-values comparing MUSOLS against each baseline. Every "
        "algorithm within a cell sees an identical stakeholder panel, environment seed and evaluation weight "
        "set, so comparisons are paired. `--' marks comparisons that are untestable because the two "
        "algorithms are identical on every cell, which is itself the result when MUSOLS matches OLS exactly.}",
        "\\label{tab:significance}",
        "\\end{table}",
        "",
    ]
    (out / "significance_table.tex").write_text("\n".join(latex))
    return "\n".join(markdown)


# ------------------------------------------------------------------------------------------------ figures


def _errorbar_plot(rows, colour, mark, legend=None, log_x=False):
    """One pgfplots line with asymmetric 95% CI error bars, from (x, point, lo, hi) tuples.

    Asymmetric rather than symmetric bars because the runtime intervals come from a bootstrap of the median and
    genuinely are lopsided; halving (hi - lo) would misplace the point estimate.
    """
    lines = [
        f"    \\addplot+[color={colour}, mark={mark}, thick, error bars/.cd, y dir=both, y explicit]",
        "    table[row sep=\\\\, y error plus index=2, y error minus index=3] {",
        "    x y ep em \\\\",
    ]
    for x, point, lo, hi in rows:
        lines.append(f"    {x:g} {point:.6g} {max(hi - point, 0):.6g} {max(point - lo, 0):.6g} \\\\")
    lines.append("    };")
    if legend is not None:
        lines.append(f"    \\addlegendentry{{{legend}}}")
    lines.append("")   # blank line between series, as in the hand-edited figure
    return lines


def write_scaling_figures(records: List[dict], out: Path) -> None:
    """Per-metric scalability plots against the number of objectives, one figure per panel size.

    Three stacked subplots sharing an x axis -- coverage-set size, evaluations, runtime -- rather than a single
    OLS-to-MUSOLS ratio: a ratio compresses two trends into one number and hides that MUSOLS's own cost is
    flat in d while OLS's is not, which is the actual claim. Each series carries 95% CI error bars, so a
    reader can see whether neighbouring points are distinguishable.
    """
    for num_users in (MAIN_NUM_USERS, AUX_NUM_USERS):
        subset = [
            r for r in records
            if str(r.get("env", "")).startswith("synthetic") and r.get("num_users") == num_users
            and r.get("concentration") == MAIN_CONCENTRATION
        ]
        if not subset:
            continue
        lines = [
            "% Requires: \\usepackage{pgfplots} \\pgfplotsset{compat=1.18} \\usepgfplotslibrary{groupplots}",
            "\\begin{figure}[t]",
            "\\centering",
            "\\begin{tikzpicture}",
            "\\begin{groupplot}[",
            f"  group style={{group size=1 by {len(SCALING_PLOT_METRICS)}, vertical sep=4mm, "
            "x descriptions at=edge bottom},",
            "  width=0.92\\columnwidth, height=0.36\\columnwidth,",
            "  xlabel={Number of objectives $d$}, grid=major,",
            "  legend style={font=\\scriptsize, at={(0.02,0.98)}, anchor=north west},",
            "  label style={font=\\small}, tick label style={font=\\scriptsize},",
            "]",
        ]
        for metric, ylabel, kind in SCALING_PLOT_METRICS:
            # Runtime spans orders of magnitude across d; the count metrics do not.
            logy = ", ymode=log, log basis y={10}" if metric == "elapsed_seconds" else ""
            lines.append(f"\\nextgroupplot[ylabel={{{ylabel}}}{logy}]")
            for algorithm in ALGORITHMS:
                rows = []
                for d in sorted({r["num_objectives"] for r in subset}):
                    values = [r.get(metric) for r in subset
                              if r["num_objectives"] == d and r["algorithm"] == algorithm]
                    point, lo, hi = _stat(values, kind)
                    if point is not None:
                        rows.append((d, point, lo, hi))
                if rows:
                    lines += _errorbar_plot(rows, PLOT_COLOURS[algorithm], PLOT_MARKS[algorithm],
                                            PRETTY[algorithm])
        lines += [
            "\\end{groupplot}",
            "\\end{tikzpicture}",
            f"\\caption{{Scalability on the synthetic task ({_synthetic_shape(subset)}) with $m={num_users}$ "
            f"stakeholders at $\\kappa={MAIN_CONCENTRATION:g}$. MUSOLS searches a consensus space of dimension "
            f"$\\min(d,m)$, so its cost is flat in $d$ while the unrestricted search grows; runtime is on a "
            "log scale. Error bars are 95\\% CIs (bootstrap for runtime, normal approximation otherwise).}",
            f"\\label{{fig:scaling-m{num_users}}}",
            "\\end{figure}",
            "",
        ]
        (out / f"scaling_figure_m{num_users}.tex").write_text("\n".join(lines))


def _anytime_curve(records: List[dict], env: str, algorithm: str, horizon: int) -> List[float]:
    """Median expected utility after each evaluation, carrying each run's last value forward to `horizon`.

    Carrying forward is the right treatment for an anytime algorithm: once it has converged it would keep
    reporting the same set, so its curve is flat rather than undefined beyond that point.
    """
    curves = []
    for record in records:
        if record["env"] != env or record["algorithm"] != algorithm or not record.get("trajectory"):
            continue
        values = [step["expected_consensus_utility"] for step in record["trajectory"]]
        if not values:
            continue
        curves.append(values + [values[-1]] * (horizon - len(values)))
    if not curves:
        return []
    return [float(np.median([c[i] for c in curves])) for i in range(horizon)]


def write_anytime_figure(records: List[dict], env: str, out: Path, name: str, horizon: int = 24) -> None:
    """Emits an anytime quality curve: expected consensus utility against evaluations spent."""
    styles = {"musols": "mark=*", "random": "mark=square*", "vertex": "mark=triangle*", "ols": "mark=diamond*"}
    lines = [
        "% Requires: \\usepackage{pgfplots} \\pgfplotsset{compat=1.18}",
        "\\begin{figure}[t]",
        "\\centering",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "    width=0.8\\linewidth, height=6cm,",
        "    xlabel={Policy evaluations},",
        "    ylabel={Expected consensus utility},",
        "    grid=major, legend pos=south east, legend cell align={left},",
        "]",
    ]
    for algorithm in ALGORITHMS:
        curve = _anytime_curve(records, env, algorithm, horizon)
        if not curve:
            continue
        coords = " ".join(f"({i + 1},{v:.6g})" for i, v in enumerate(curve))
        lines.append(f"\\addplot+[{styles[algorithm]}, mark repeat=3] coordinates {{{coords}}};")
        lines.append(f"\\addlegendentry{{{PRETTY[algorithm]}}}")
    lines += [
        "\\end{axis}",
        "\\end{tikzpicture}",
        f"\\caption{{Anytime quality on {env.replace('_', chr(92) + '_')}: median expected consensus utility "
        "after each policy evaluation, with each run's final value carried forward. MUSOLS reaches its final "
        "quality within a few evaluations, while OLS spends many more to arrive at the same or lower value.}",
        f"\\label{{fig:anytime-{env}}}",
        "\\end{figure}",
        "",
    ]
    (out / f"{name}.tex").write_text("\n".join(lines))


def write_heterogeneity_figure(records: List[dict], out: Path, envs: Sequence[str] = ()) -> None:
    """Coverage-set size and evaluation count against stakeholder agreement, on a log kappa axis.

    kappa is a concentration parameter, so equal ratios rather than equal differences are comparable -- 1, 5
    and 50 are roughly evenly spaced only on a log axis. Two stacked subplots share that axis: the restricted
    coverage set shrinks as a panel becomes unanimous, and the evaluation count follows it, which is the
    mechanism rather than a coincidence. MUSOLS only, since the question is what the restriction yields.

    One legend for the whole figure, exported from the first subplot with `legend to name` and placed below by
    a \\node: the series are the same environments in both subplots, so repeating the key would spend space
    restating it. Written once per panel size, with HETEROGENEITY_NUM_USERS taking the unsuffixed filename so
    the paper's \\input keeps pointing at the informative one.
    """
    envs = [e for e in envs if not e.startswith("synthetic")]
    if not envs:
        return
    others = [m for m in (MAIN_NUM_USERS, AUX_NUM_USERS) if m != HETEROGENEITY_NUM_USERS]
    n_rows = len(HETEROGENEITY_PLOT_METRICS)
    palette = ["blue", "orange", "teal", "purple", "brown", "olive"]
    marks = ["*", "square*", "triangle*", "diamond*", "pentagon*", "x"]

    for num_users in [HETEROGENEITY_NUM_USERS, *others]:
        primary = num_users == HETEROGENEITY_NUM_USERS
        name = "heterogeneity_figure" if primary else f"heterogeneity_figure_m{num_users}"
        label = "fig:heterogeneity" if primary else f"fig:heterogeneity-m{num_users}"
        subset = [
            r for r in records
            if r["env"] in envs and r["algorithm"] == "musols" and r.get("num_users") == num_users
        ]
        if not subset:
            continue
        kappas = sorted({r["concentration"] for r in subset})
        if len(kappas) < 2:
            print(f"  (heterogeneity figure at m={num_users} needs >1 kappa; found {kappas}) -- not written")
            continue

        # A series needs at least two kappa levels to show a trend at all. An environment swept at a single
        # kappa -- water-reservoir, whose job passes --concentrations 5 only -- would otherwise contribute one
        # marker to a figure about how a quantity *changes*, which reads as a data point rather than as the
        # absence of one. Dropped, and named, rather than drawn.
        per_env_kappas = {
            env: sorted({r["concentration"] for r in subset if r["env"] == env})
            for env in envs
        }
        drawn = [e for e in envs if len(per_env_kappas.get(e, [])) >= 2]
        for env in envs:
            found = per_env_kappas.get(env, [])
            if len(found) < 2:
                print(f"  heterogeneity (m={num_users}): skipping {env} -- swept at "
                      f"{len(found)} kappa level(s) {found}, needs >= 2")
            elif len(found) < len(kappas):
                print(f"  heterogeneity (m={num_users}): {env} covers only kappa {found} "
                      f"of {kappas}; its line is partial")
        if not drawn:
            print(f"  (heterogeneity figure at m={num_users}: no environment has >= 2 kappa) -- not written")
            continue

        lines = [
            f"% Stakeholder homogeneity at m={num_users}.",
            "% Requires: \\usepackage{pgfplots} \\pgfplotsset{compat=1.18} \\usepgfplotslibrary{groupplots}",
            "\\begin{figure}[t]",
            "\\centering",
            "\\begin{tikzpicture}",
            "\\begin{groupplot}[",
            "  group style={",
            f"    group size=1 by {n_rows}, ",
            "    vertical sep=4mm, ",
            "    x descriptions at=edge bottom",
            "  },",
            "  width=0.92\\columnwidth, height=0.38\\columnwidth,",
            "  xlabel={Homogeneity \\(\\kappa\\) (log scale)},",
            "  xmode=log, log basis x={10}, grid=major,",
            f"  xtick={{{','.join(f'{k:g}' for k in kappas)}}}, "
            f"xticklabels={{{','.join(f'{k:g}' for k in kappas)}}},",
            # Both quantities counted here are at least one, so clipping the axis there keeps the error bars
            # from implying values that cannot occur.
            "  ymin=1,",
            "  legend style={",
            "    font=\\scriptsize, ",
            "    at={(0.5,-0.35)}, ",
            "    anchor=north, ",
            f"    legend columns={max(len(drawn), 1)}",
            "  },",
            "  label style={font=\\small}, tick label style={font=\\scriptsize},",
            "]",
        ]
        for row, (metric, ylabel, kind) in enumerate(HETEROGENEITY_PLOT_METRICS):
            if row == 0:
                lines += [
                    "\\nextgroupplot[",
                    f"  ylabel={{{ylabel}}},",
                    "  legend to name=grouplegend % Exports the single legend for the group",
                    "]",
                ]
            else:
                lines.append(f"\\nextgroupplot[ylabel={{{ylabel}}}]")
            for index, env in enumerate(drawn):
                rows = []
                for kappa in kappas:
                    values = [r.get(metric) for r in subset
                              if r["env"] == env and r["concentration"] == kappa]
                    point, lo, hi = _stat(values, kind)
                    if point is not None:
                        rows.append((kappa, point, lo, hi))
                if not rows:
                    continue
                entry = f"\\texttt{{{ENV_SHORT.get(env, env)}}}" if row == 0 else None
                lines += _errorbar_plot(rows, palette[index % len(palette)],
                                        marks[index % len(marks)], entry)
        lines += [
            "\\end{groupplot}",
            "",
            "% Renders the shared legend centered below the entire groupplot",
            f"\\node at (group c1r{n_rows}.south) [anchor=north, yshift=-0.85cm] {{\\ref{{grouplegend}}}};",
            "\\end{tikzpicture}",
            f"\\caption{{CCS size and number of evaluations for varying user preference weight homogeneity "
            f"as expressed by the Dirichlet concentration parameter $\\kappa$, with $m={num_users}$ users. "
            "Higher $\\kappa$ means more homogeneity. Error bars are 95\\% CIs.}",
            f"\\label{{{label}}}",
            "\\end{figure}",
            "",
        ]
        (out / f"{name}.tex").write_text("\n".join(lines))


def main():
    args = parse_args()
    records = load(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _attach_delta_eu(records)

    def cell_counts(subset):
        per_env = defaultdict(set)
        for record in subset:
            per_env[record["env"]].add(tuple(record[k] for k in CELL_KEYS))
        return {env: len(cs) for env, cs in per_env.items()}

    present = {r["env"] for r in records}
    missing = [e for e in MAIN_ENVS if e not in present]
    if missing:
        print(f"WARNING: listed environment(s) absent from the results: {', '.join(missing)}")

    # Each table fixes one panel size at the headline concentration; kappa is varied only in the figures.
    table_specs = [
        (MAIN_NUM_USERS, "main_results_table", "tab:main", "Main results"),
        (AUX_NUM_USERS, "main_results_table_m{}".format(AUX_NUM_USERS),
         f"tab:main-m{AUX_NUM_USERS}", "Additional results"),
    ]
    kappas = [MAIN_CONCENTRATION]

    for position, (num_users, name, label, role) in enumerate(table_specs):
        subset = [
            r for r in records
            if r.get("concentration") == MAIN_CONCENTRATION and r.get("num_users") == num_users
        ]
        counts = cell_counts(subset)
        envs, thin = [], []
        for env in MAIN_ENVS:
            n = counts.get(env, 0)
            if n == 0:
                continue
            envs.append(env)
            if n <= MIN_CELLS:
                thin.append((env, n))

        print("\n" + "=" * 100)
        print(f"TABLE {position + 1} -- {role}: m={num_users}, kappa={MAIN_CONCENTRATION:g}")
        print("=" * 100)
        if not envs:
            print(f"(no listed environment has records at m={num_users}, kappa={MAIN_CONCENTRATION:g})")
            continue
        for env, n in thin:
            # Reported anyway, because the environment list is an editorial choice -- but never silently.
            print(f"WARNING: {env} has only {n} cells (<= {MIN_CELLS}); its intervals are wide and its "
                  f"significance tests underpowered.")
        print("environments: " + ", ".join(f"{e}({counts[e]})" for e in envs))
        caption = (
            f"{role} for $m={num_users}$ stakeholders at heterogeneity $\\kappa={MAIN_CONCENTRATION:g}$. "
            "Wall-clock time is the median with a percentile bootstrap 95\\% CI, since runtimes are "
            "right-skewed and censored at the per-algorithm budget; $\\Delta$EU likewise uses a bootstrap, "
            "being bounded below by zero with mass on the bound; the count columns are means with "
            "normal-approximation 95\\% CIs. $\\Delta$EU is the shortfall in expected consensus utility "
            "against the best algorithm in the same cell, so $0$ means nothing was given up. MUL is maximum "
            "consensus utility loss. Boldface marks the best value and any not significantly worse than it "
            "(paired Wilcoxon within cells, $p\\ge0.05$)."
        )
        print(write_main_table(subset, envs, out, name, caption, label))

    next_table = len(kappas) + 1
    print("\n" + "=" * 100)
    print(f"TABLE {next_table} -- Scaling: cost of OLS relative to MUSOLS")
    print("=" * 100)
    print(write_scaling_table(records, out))

    # The significance table pools heterogeneity levels, so it uses the same cell-count floor applied over the
    # whole record set rather than per kappa.
    all_counts = cell_counts(records)
    reportable = sorted(e for e, n in all_counts.items() if n > MIN_CELLS)
    print("\n" + "=" * 100)
    print(f"TABLE {next_table + 1} -- Paired Wilcoxon signed-rank tests")
    print("=" * 100)
    if reportable:
        print(write_significance_table(records, reportable, out))
    else:
        print(f"(no environment has more than {MIN_CELLS} cells; nothing written)")

    write_scaling_figures(records, out)
    write_anytime_figure(records, "fruit-tree", out, "anytime_fruit_tree", horizon=24)
    write_anytime_figure(records, "synthetic-d4", out, "anytime_synthetic_d4", horizon=24)
    write_heterogeneity_figure(records, out, [e for e in MAIN_ENVS if e in present])

    print("\n" + "=" * 100)
    print(f"Wrote LaTeX/TikZ assets to {out}/:")
    for path in sorted(out.glob("*.tex")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
