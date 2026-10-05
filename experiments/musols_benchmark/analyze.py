"""Turns study result files into the tables a paper reports.

Reads one or more JSON Lines files written by `run_study.py` and aggregates them. Two deliberate choices about
how results are summarized:

  - **Median and IQR, not mean and standard deviation.** Several of the quantities here (wall-clock time,
    evaluation counts, CCS size) are strongly skewed -- a single cell where OLS hits its timeout dominates a
    mean -- and none are normally distributed. Medians with an interquartile range describe these honestly.
  - **Paired, non-parametric significance tests.** Every algorithm in a cell sees the identical stakeholder
    panel, environment seed and evaluation weights, so comparisons are naturally paired. Pairing is matched on
    the full cell key and compared with a Wilcoxon signed-rank test, which assumes neither normality nor equal
    variances.

A note on utility loss: it is only an absolute quantity when measured against a brute-forced ground-truth
restricted CCS, which exists only for enumerable environments. Elsewhere the reference is the union of all
algorithms' returned sets, which makes the metric comparative and mechanically favours whichever algorithm
contributed most of the union. Records carry `reference_is_ground_truth` and this script reports it, so the
two are never silently mixed.

Examples:
    python experiments/musols_benchmark/analyze.py results/*.jsonl
    python experiments/musols_benchmark/analyze.py results/scaling.jsonl --group-by env num_users --latex
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import wilcoxon


# Same order and naming as make_paper_assets.py, so a terminal table and the paper's table can be read
# against each other. The two baselines are ablations of MUSOLS: "-consensus" drops the consensus search and
# solves each stakeholder's own weight; "-search" drops the prioritized corner-weight search for uniform
# sampling of alpha, keeping the restriction to Omega_W.
ALGORITHM_ORDER = ("ols", "musols", "vertex", "random")
PRETTY = {"ols": "OLS", "musols": "MUSOLS", "vertex": "-consensus", "random": "-search"}
# The cell coordinates that identify one paired comparison across algorithms.
CELL_KEYS = ("env", "seed", "num_users", "concentration")
METRICS = (
    ("ccs_size", "|CCS|", False),
    ("num_evaluated", "evals", False),
    ("elapsed_seconds", "time(s)", False),
    ("expected_consensus_utility", "EU", True),
    ("max_consensus_utility_loss", "MUL", False),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=str, nargs="+", help="JSON Lines files written by run_study.py.")
    parser.add_argument(
        "--group-by",
        type=str,
        nargs="+",
        default=["env", "concentration"],
        help=(
            "Record fields to group rows by. Defaults to env and concentration: stakeholder heterogeneity "
            "changes the achievable restricted coverage set substantially (at d=4 the median |CCS_W| is 2 at "
            "kappa=1 but 1 at kappa=5, and 72% of kappa=50 panels admit a single policy), so pooling the "
            "kappa levels averages over qualitatively different regimes. Add num_users to split by panel size "
            "too, or pass just `env` to pool everything."
        ),
    )
    parser.add_argument("--latex", action="store_true", help="Emit LaTeX tabular rows instead of markdown.")
    parser.add_argument(
        "--baseline",
        type=str,
        default="musols",
        help="Algorithm that the significance tests compare every other algorithm against.",
    )
    return parser.parse_args()


def load(paths: Sequence[str]) -> List[dict]:
    """Loads every record from the given JSON Lines files."""
    records = []
    for path in paths:
        with Path(path).open() as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def _median_iqr(values: Sequence[float]) -> str:
    """Formats a sample as median with its interquartile range."""
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return "-"
    q25, q75 = np.percentile(arr, [25, 75])
    return f"{np.median(arr):.3g} [{q25:.3g}, {q75:.3g}]"


def _paired(records: List[dict], baseline: str, algorithm: str, metric: str):
    """Extracts the paired samples of `metric` for two algorithms over the cells where both ran."""
    by_cell: Dict[tuple, Dict[str, float]] = defaultdict(dict)
    for record in records:
        by_cell[tuple(record[k] for k in CELL_KEYS)][record["algorithm"]] = record[metric]
    pairs = [(cell[baseline], cell[algorithm]) for cell in by_cell.values() if baseline in cell and algorithm in cell]
    pairs = [(a, b) for a, b in pairs if not (np.isnan(a) or np.isnan(b))]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def significance(records: List[dict], baseline: str, algorithm: str, metric: str) -> str:
    """Wilcoxon signed-rank p-value for one metric, baseline vs algorithm, over paired cells."""
    left, right = _paired(records, baseline, algorithm, metric)
    if left.size < 3 or np.allclose(left, right):
        return "-"  # too few pairs, or the two are identical everywhere, so there is nothing to test
    try:
        return f"{wilcoxon(left, right).pvalue:.3g}"
    except ValueError:
        return "-"


def summarize(records: List[dict], group_by: Sequence[str], latex: bool) -> None:
    """Prints one summary table per group, with a row per algorithm."""
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for record in records:
        groups[tuple(record[k] for k in group_by)].append(record)

    for key in sorted(groups, key=lambda k: tuple(str(v) for v in k)):
        block = groups[key]
        heading = ", ".join(f"{field}={value}" for field, value in zip(group_by, key))
        n_cells = len({tuple(r[k] for k in CELL_KEYS) for r in block})
        ground_truth = {r["reference_is_ground_truth"] for r in block}
        reference = (
            "ground truth" if ground_truth == {True} else ("union-of-algorithms" if ground_truth == {False} else "mixed")
        )
        print(f"\n### {heading}  (n={n_cells} cells, MUL reference: {reference})")

        columns = ["algorithm"] + [label for _, label, _ in METRICS] + ["converged"]
        rows = []
        for algorithm in ALGORITHM_ORDER:
            subset = [r for r in block if r["algorithm"] == algorithm]
            if not subset:
                continue
            cells = [_median_iqr([r[m] for r in subset]) for m, _, _ in METRICS]
            converged = f"{sum(bool(r['converged']) for r in subset)}/{len(subset)}"
            rows.append([PRETTY.get(algorithm, algorithm)] + cells + [converged])

        if latex:
            print("\\begin{tabular}{l" + "r" * (len(columns) - 1) + "}")
            print("\\toprule")
            print(" & ".join(columns) + " \\\\")
            print("\\midrule")
            for row in rows:
                print(" & ".join(str(c) for c in row) + " \\\\")
            print("\\bottomrule")
            print("\\end{tabular}")
        else:
            print("| " + " | ".join(columns) + " |")
            print("|" + "|".join("---" for _ in columns) + "|")
            for row in rows:
                print("| " + " | ".join(str(c) for c in row) + " |")


def significance_table(records: List[dict], baseline: str, group_by: Sequence[str]) -> None:
    """Prints paired Wilcoxon p-values against the baseline, grouped exactly as the summary tables are.

    Grouping matters here rather than being cosmetic: pooling stakeholder-heterogeneity levels tests a mixture
    of regimes in which the achievable restricted coverage set differs in size, so a pooled p-value can be
    driven by whichever level happens to contribute most cells. Using the same grouping as `summarize` also
    means every p-value below corresponds to exactly one summary block above it.
    """
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for record in records:
        groups[tuple(record[k] for k in group_by)].append(record)

    others = [a for a in ALGORITHM_ORDER if a != baseline]
    print(f"\n### Paired Wilcoxon signed-rank vs {PRETTY.get(baseline, baseline)} "
          "(p-values; '-' = untestable or identical)")
    header = list(group_by) + ["n", "comparison"] + [label for _, label, _ in METRICS]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")

    for key in sorted(groups, key=lambda k: tuple(str(v) for v in k)):
        block = groups[key]
        n_cells = len({tuple(r[k] for k in CELL_KEYS) for r in block})
        present = {r["algorithm"] for r in block}
        for algorithm in others:
            if algorithm not in present:
                continue
            cells = [significance(block, baseline, algorithm, metric) for metric, _, _ in METRICS]
            label = " | ".join(str(v) for v in key)
            comparison = f"{PRETTY.get(baseline, baseline)} vs {PRETTY.get(algorithm, algorithm)}"
            print(f"| {label} | {n_cells} | {comparison} | " + " | ".join(cells) + " |")


def main():
    args = parse_args()
    records = load(args.results)
    if not records:
        print("No records found.")
        return
    print(f"Loaded {len(records)} records from {len(args.results)} file(s).")
    print(f"Environments: {sorted({r['env'] for r in records})}")
    summarize(records, args.group_by, args.latex)
    significance_table(records, args.baseline, args.group_by)


if __name__ == "__main__":
    main()
