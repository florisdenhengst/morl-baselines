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


ALGORITHMS = ("musols", "random", "vertex", "ols")
PRETTY = {"musols": "MUSOLS", "random": "Random-$\\Omega_W$", "vertex": "Vertex-only", "ols": "OLS"}
PRETTY_MD = {"musols": "MUSOLS", "random": "Random-Ω_W", "vertex": "Vertex-only", "ols": "OLS"}
CELL_KEYS = ("env", "seed", "num_users", "concentration")
WELL_POWERED = ("deep-sea-treasure", "fruit-tree", "resource-gathering", "synthetic-d2", "synthetic-d3", "synthetic-d4")


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


def _median_iqr(values: Sequence[float]) -> str:
    """Median with interquartile range; the skew in these quantities makes mean/sd misleading."""
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return "--"
    q25, q75 = np.percentile(arr, [25, 75])
    return f"{_fmt(float(np.median(arr)))} [{_fmt(float(q25))}, {_fmt(float(q75))}]"


def _by_cell(records: List[dict]) -> Dict[tuple, Dict[str, dict]]:
    """Indexes records by cell coordinates, so algorithms can be compared pairwise within a cell."""
    cells: Dict[tuple, Dict[str, dict]] = defaultdict(dict)
    for record in records:
        cells[tuple(record[k] for k in CELL_KEYS)][record["algorithm"]] = record
    return cells


# --------------------------------------------------------------------------------------- main results table


def main_results_rows(records: List[dict], envs: Sequence[str]) -> List[List[str]]:
    """Builds the per-environment, per-algorithm summary rows shared by the markdown and LaTeX tables."""
    rows = []
    for env in envs:
        block = [r for r in records if r["env"] == env]
        if not block:
            continue
        n_cells = len({tuple(r[k] for k in CELL_KEYS) for r in block})
        for index, algorithm in enumerate(ALGORITHMS):
            subset = [r for r in block if r["algorithm"] == algorithm]
            if not subset:
                continue
            rows.append(
                [
                    f"{env} (n={n_cells})" if index == 0 else "",
                    algorithm,
                    _median_iqr([r["ccs_size"] for r in subset]),
                    _median_iqr([r["num_evaluated"] for r in subset]),
                    _median_iqr([r["elapsed_seconds"] for r in subset]),
                    _median_iqr([r["expected_consensus_utility"] for r in subset]),
                    _median_iqr([r["max_consensus_utility_loss"] for r in subset]),
                ]
            )
    return rows


def write_main_table(records: List[dict], envs: Sequence[str], out: Path, name: str, caption: str, label: str) -> str:
    """Writes the LaTeX main-results table and returns its markdown twin."""
    rows = main_results_rows(records, envs)
    headers = ["Environment", "Algorithm", "|CCS|", "Evaluations", "Time (s)", "EU", "MUL"]

    latex = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Environment & Algorithm & $|\\mathcal{C}_W|$ & Evals. & Time (s) & EU $\\uparrow$ & MUL $\\downarrow$ \\\\",
        "\\midrule",
    ]
    previous_env = None
    for row in rows:
        if row[0] and previous_env is not None:
            latex.append("\\midrule")
        previous_env = row[0] or previous_env
        env_cell = row[0].replace("_", "\\_") if row[0] else ""
        cells = [env_cell, PRETTY[row[1]]] + [c.replace("[", "{\\scriptsize [").replace("]", "]}") for c in row[2:]]
        latex.append(" & ".join(cells) + " \\\\")
    latex += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}", ""]
    (out / f"{name}.tex").write_text("\n".join(latex))

    markdown = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        markdown.append("| " + " | ".join([row[0], PRETTY_MD[row[1]]] + row[2:]) + " |")
    return "\n".join(markdown)


# ------------------------------------------------------------------------------------------- scaling table


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
        "\\caption{Cost of OLS relative to MUSOLS on the synthetic task ($N=20$ attainable payoffs, sphere "
        "geometry), median over cells. The wall-clock ratio grows superlinearly in the number of objectives, "
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
        "single-decision MOMDP with $N=20$ attainable payoffs in sphere geometry, so that $|\\mathcal{C}|=N$ "
        "by construction; median over cells, log-scaled ordinate.}",
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


def write_heterogeneity_figure(records: List[dict], out: Path) -> None:
    """Emits the effect of stakeholder agreement on the size of the returned restricted coverage set."""
    envs = [e for e in WELL_POWERED if not e.startswith("synthetic")]
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

    present = {r["env"] for r in records}
    well_powered = [e for e in WELL_POWERED if e in present]
    indicative = sorted(e for e in present if e not in WELL_POWERED)

    print("=" * 100)
    print("TABLE 1 -- Main results, well-powered environments")
    print("=" * 100)
    print(
        write_main_table(
            records,
            well_powered,
            out,
            "main_results_table",
            "Main results on the exact-solver and synthetic environments, where every configuration has tens "
            "of cells and utility loss is measured against a brute-forced ground-truth restricted CCS. "
            "Median [interquartile range]. EU is expected consensus utility (higher is better), MUL is "
            "maximum consensus utility loss (lower is better).",
            "tab:main",
        )
    )

    print("\n" + "=" * 100)
    print("TABLE 2 -- Indicative results, deep-RL environments (underpowered; see caveats)")
    print("=" * 100)
    print(
        write_main_table(
            records,
            indicative,
            out,
            "indicative_results_table",
            "Indicative results on the deep-RL environments. These are reported separately because only a "
            "handful of cells were affordable and several ran at reduced training budgets, so differences "
            "here are not statistically resolvable; utility loss is measured against the union of all "
            "algorithms' returned sets rather than a ground truth. Median [interquartile range].",
            "tab:indicative",
        )
    )

    print("\n" + "=" * 100)
    print("TABLE 3 -- Scaling: cost of OLS relative to MUSOLS")
    print("=" * 100)
    print(write_scaling_table(records, out))

    print("\n" + "=" * 100)
    print("TABLE 4 -- Paired Wilcoxon signed-rank tests")
    print("=" * 100)
    print(write_significance_table(records, well_powered, out))

    write_scaling_figure(records, out)
    write_anytime_figure(records, "fruit-tree", out, "anytime_fruit_tree", horizon=24)
    write_anytime_figure(records, "synthetic-d4", out, "anytime_synthetic_d4", horizon=24)
    write_heterogeneity_figure(records, out)

    print("\n" + "=" * 100)
    print(f"Wrote LaTeX/TikZ assets to {out}/:")
    for path in sorted(out.glob("*.tex")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
