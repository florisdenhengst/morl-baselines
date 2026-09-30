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


# Ordered so the table builds up to the proposed method: naive baseline, undirected baseline, the
# unrestricted algorithm MUSOLS specialises, then MUSOLS itself.
ALGORITHMS = ("vertex", "random", "ols", "musols")
PRETTY = {"musols": "MUSOLS", "random": "Random-$\\Omega_W$", "vertex": "Vertex-only", "ols": "OLS"}
PRETTY_MD = {"musols": "MUSOLS", "random": "Random-Ω_W", "vertex": "Vertex-only", "ols": "OLS"}
CELL_KEYS = ("env", "seed", "num_users", "concentration")
# Environments are split by how much evidence they carry, measured from the records rather than listed by
# name: a hardcoded list silently misfiled synthetic-d5..d8 (600 cells each, ground-truth utility loss) as
# "indicative", and would have done the same to every newly added environment.
# Environments with this many cells or fewer are excluded from the assets entirely rather than reported in a
# separate "indicative" table: at that sample size no comparison is resolvable, and publishing the rows invites
# them to be read as results.
MIN_CELLS = 100
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


def _fmt_stat(point, lo, hi, bold: bool = False) -> str:
    if point is None:
        return "--"
    body = f"{_fmt(point)} [{_fmt(lo)}, {_fmt(hi)}]"
    return f"\\textbf{{{_fmt(point)}}} [{_fmt(lo)}, {_fmt(hi)}]" if bold else body


def _fmt_stat_md(point, lo, hi, bold: bool = False) -> str:
    if point is None:
        return "--"
    head = f"**{_fmt(point)}**" if bold else _fmt(point)
    return f"{head} [{_fmt(lo)}, {_fmt(hi)}]"


# Metrics reported in the main table. `higher_better` drives which end counts as best for boldface;
# `kind` picks the interval; ccs/evals/time are costs, delta-EU and MUL are quality.
TABLE_METRICS = (
    ("ccs_size", "$|\\mathcal{C}_W|$", "|CCS|", False, "mean"),
    ("num_evaluated", "Evals.", "evals", False, "mean"),
    ("elapsed_seconds", "Time (s)", "time(s)", False, "median_boot"),
    ("delta_eu", "$\\Delta$EU $\\downarrow$", "dEU", False, "mean_boot"),
    ("max_consensus_utility_loss", "MUL $\\downarrow$", "MUL", False, "mean"),
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


def main_results_rows(records: List[dict], envs: Sequence[str], markdown: bool = False) -> List[List[str]]:
    """Per-environment, per-algorithm rows with 95% intervals and significance-aware boldface."""
    fmt = _fmt_stat_md if markdown else _fmt_stat
    rows = []
    for env in envs:
        block = [r for r in records if r["env"] == env]
        if not block:
            continue
        n_cells = len({tuple(r[k] for k in CELL_KEYS) for r in block})
        bold = {m: _bold_set(block, m, hi) for m, _, _, hi, _ in TABLE_METRICS}
        for index, algorithm in enumerate(ALGORITHMS):
            subset = [r for r in block if r["algorithm"] == algorithm]
            if not subset:
                continue
            cells = []
            for metric, _, _, _, kind in TABLE_METRICS:
                point, lo, hi = _stat([r.get(metric) for r in subset], kind)
                cells.append(fmt(point, lo, hi, bold=algorithm in bold[metric]))
            rows.append([f"{env} (n={n_cells})" if index == 0 else "", algorithm] + cells)
    return rows


def write_main_table(records: List[dict], envs: Sequence[str], out: Path, name: str, caption: str, label: str) -> str:
    """Writes the LaTeX main-results table and returns its markdown twin."""
    latex_head = " & ".join(["Environment", "Algorithm"] + [tex for _, tex, _, _, _ in TABLE_METRICS])
    latex = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{ll" + "r" * len(TABLE_METRICS) + "}",
        "\\toprule",
        latex_head + " \\\\",
        "\\midrule",
    ]
    previous_env = None
    for row in main_results_rows(records, envs):
        if row[0] and previous_env is not None:
            latex.append("\\midrule")
        previous_env = row[0] or previous_env
        env_cell = row[0].replace("_", "\\_") if row[0] else ""
        cells = [env_cell, PRETTY[row[1]]] + [
            c.replace("[", "{\\scriptsize [").replace("]", "]}") for c in row[2:]
        ]
        latex.append(" & ".join(cells) + " \\\\")
    latex += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
              "\\end{table}", ""]
    (out / f"{name}.tex").write_text("\n".join(latex))

    headers = ["Environment", "Algorithm"] + [md for _, _, md, _, _ in TABLE_METRICS]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in main_results_rows(records, envs, markdown=True):
        lines.append("| " + " | ".join([row[0], PRETTY_MD[row[1]]] + row[2:]) + " |")
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


def write_scaling_figure(records: List[dict], out: Path) -> None:
    """Emits the headline scaling figure: OLS-to-MUSOLS cost ratios against the number of objectives."""
    ratios = scaling_ratios(records)
    series = {
        "Wall-clock time": ("time", "mark=*"),
        "Evaluations": ("evals", "mark=square*"),
        "$|\\mathcal{C}|$": ("ccs", "mark=triangle*"),
    }
    lines = [
        "% Requires: \\usepackage{pgfplots} \\pgfplotsset{compat=1.18}",
        "\\begin{figure}[t]",
        "\\centering",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "    width=0.8\\linewidth, height=6cm,",
        "    xlabel={Number of objectives $d$},",
        "    ylabel={Cost of OLS relative to MUSOLS},",
        "    ymode=log, log basis y={10},",
        "    xtick={" + ",".join(str(d) for d in ratios) + "},",
        "    grid=major, legend pos=north west, legend cell align={left},",
        "]",
    ]
    for label, (key, style) in series.items():
        coords = " ".join(f"({d},{metrics[key]:.4g})" for d, metrics in ratios.items())
        lines.append(f"\\addplot+[{style}] coordinates {{{coords}}};")
        lines.append(f"\\addlegendentry{{{label}}}")
    lines += [
        "\\end{axis}",
        "\\end{tikzpicture}",
        "\\caption{MUSOLS's advantage over OLS grows superlinearly in the number of objectives. Synthetic "
        f"single-decision MOMDP ({_synthetic_shape(records)}), "
        "median over cells, log-scaled ordinate.}",
        "\\label{fig:scaling}",
        "\\end{figure}",
        "",
    ]
    (out / "scaling_figure.tex").write_text("\n".join(lines))


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
    """Emits the effect of stakeholder agreement on the size of the returned restricted coverage set.

    Takes the environment list from the caller, which derives it from cell counts, rather than consulting a
    hardcoded roster. Synthetic tasks are excluded because their heterogeneity response is already covered by
    the scaling figure, and including seven of them would swamp the real environments.
    """
    envs = [e for e in envs if not e.startswith("synthetic")]
    if not envs:
        return
    per_env = defaultdict(dict)
    for record in records:
        if record["env"] not in envs or record["algorithm"] != "musols":
            continue
        per_env[record["env"]].setdefault(record["concentration"], []).append(record["ccs_size"])
    kappas = sorted({k for values in per_env.values() for k in values})
    lines = [
        "% Requires: \\usepackage{pgfplots} \\pgfplotsset{compat=1.18}",
        "\\begin{figure}[t]",
        "\\centering",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "    width=0.8\\linewidth, height=6cm,",
        "    xlabel={Stakeholder agreement $\\kappa$ (larger is more unanimous)},",
        "    ylabel={$|\\mathcal{C}_W|$ returned by MUSOLS},",
        "    xmode=log, log basis x={10},",
        "    xtick={" + ",".join(f"{k:g}" for k in kappas) + "},",
        "    xticklabels={" + ",".join(f"{k:g}" for k in kappas) + "},",
        "    grid=major, legend pos=north east, legend cell align={left},",
        "]",
    ]
    for env in envs:
        if env not in per_env:
            continue
        coords = " ".join(f"({k:g},{np.median(per_env[env][k]):.4g})" for k in kappas if k in per_env[env])
        lines.append(f"\\addplot+[mark=*] coordinates {{{coords}}};")
        lines.append(f"\\addlegendentry{{{env.replace('_', chr(92) + '_')}}}")
    lines += [
        "\\end{axis}",
        "\\end{tikzpicture}",
        "\\caption{The restricted coverage set shrinks as the stakeholder panel becomes more unanimous. "
        "$\\kappa$ is the Dirichlet concentration used to sample the panel around a uniformly drawn anchor; "
        "median over seeds and panel sizes.}",
        "\\label{fig:heterogeneity}",
        "\\end{figure}",
        "",
    ]
    (out / "heterogeneity_figure.tex").write_text("\n".join(lines))


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

    kappas = sorted({r["concentration"] for r in records if r.get("concentration") is not None})
    if MAIN_CONCENTRATION in kappas:
        kappas = [MAIN_CONCENTRATION] + [k for k in kappas if k != MAIN_CONCENTRATION]
    else:
        print(f"WARNING: no records at kappa={MAIN_CONCENTRATION:g}; reporting the levels that are present.")

    dropped_any = {}
    for position, kappa in enumerate(kappas):
        subset = [r for r in records if r.get("concentration") == kappa]
        counts = cell_counts(subset)
        envs = sorted(e for e, n in counts.items() if n > MIN_CELLS)
        dropped = {e: n for e, n in counts.items() if n <= MIN_CELLS}
        dropped_any.update(dropped)

        headline = position == 0
        name = "main_results_table" if headline else f"main_results_table_k{kappa:g}"
        label = "tab:main" if headline else f"tab:main-k{kappa:g}"
        role = "Main results" if headline else "Auxiliary results"
        heading = f"TABLE {position + 1} -- {role} at kappa={kappa:g}"

        print("\n" + "=" * 100)
        print(heading)
        print("=" * 100)
        if not envs:
            print(f"(no environment has more than {MIN_CELLS} cells at kappa={kappa:g}; nothing written)")
            continue
        print(f"environments: " + ", ".join(f"{e}({counts[e]})" for e in envs))
        caption = (
            f"{role} at stakeholder heterogeneity $\\kappa={kappa:g}$. Only environments with more than "
            f"{MIN_CELLS} sampled stakeholder panels are reported. Wall-clock time is the median with a "
            "percentile bootstrap 95\\% CI, since runtimes are right-skewed and censored at the per-algorithm "
            "budget; $\\Delta$EU likewise uses a bootstrap, being bounded below by zero with mass on the "
            "bound; the count columns are means with normal-approximation 95\\% CIs. $\\Delta$EU is the "
            "shortfall in expected consensus utility against the best algorithm in the same cell, so $0$ means "
            "nothing was given up. MUL is maximum consensus utility loss. Boldface marks the best value and any "
            "not significantly worse than it (paired Wilcoxon within cells, $p\\ge0.05$)."
        )
        print(write_main_table(subset, envs, out, name, caption, label))

    if dropped_any:
        print("\n" + "-" * 100)
        print(f"Excluded (<= {MIN_CELLS} cells, not reported at any kappa where they were thin):")
        for env, n in sorted(dropped_any.items()):
            print(f"  {env}: {n} cells")

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

    write_scaling_figure(records, out)
    write_anytime_figure(records, "fruit-tree", out, "anytime_fruit_tree", horizon=24)
    write_anytime_figure(records, "synthetic-d4", out, "anytime_synthetic_d4", horizon=24)
    write_heterogeneity_figure(records, out, reportable)

    print("\n" + "=" * 100)
    print(f"Wrote LaTeX/TikZ assets to {out}/:")
    for path in sorted(out.glob("*.tex")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
