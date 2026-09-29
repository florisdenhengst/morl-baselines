#!/usr/bin/env bash
# Merges the per-shard result files into one file per sweep, then reports what arrived.
#
#   ./collect.sh                        # merge every sweep into results/merged/
#   ./collect.sh --check                # report completeness without writing anything
#   ./collect.sh resource-gathering      # merge only the named sweep(s)
#   ./collect.sh --check resource-gathering dst   # check a subset without writing anything
#
# Each array task writes its own file precisely so that concurrent tasks never interleave writes: a record
# carrying an anytime trajectory is far larger than the atomic-append limit, so appending from many jobs to a
# shared file would corrupt it. Merging is therefore a separate, deliberate step -- run this before analyze.py
# or make_paper_assets.py, not after pointing them at results/<sweep>/shard_*.jsonl directly, since only this
# script deduplicates and drops truncated tails.
#
# Because output files are opened in append mode and shard filenames are deterministic, resubmitting a sweep
# against the same results/<name>/ directory -- e.g. while testing the pipeline -- duplicates every cell that
# submission re-ran. Merging therefore also deduplicates: within each (algorithm, env, seed, num_users,
# concentration) group it keeps only the record with the latest timestamp_utc, and warns if a group's records
# disagree on git_commit, since that means the "duplicates" are not harmless reruns of identical code and
# picking the latest one is a judgment call, not a safe default.
set -euo pipefail

CHECK_ONLY=0
[[ ${1:-} == "--check" ]] && { CHECK_ONLY=1; shift; }
WANTED=("$@")   # sweep names to restrict to; empty means every sweep under results/

wanted() {
    [[ ${#WANTED[@]} -eq 0 ]] && return 0
    for w in "${WANTED[@]}"; do [[ $w == "$1" ]] && return 0; done
    return 1
}

cd "$(dirname "$0")"
REPO_ROOT=${MUSOLS_REPO:-$(cd ../../.. && pwd)}
RESULTS="$REPO_ROOT/results"
MERGED="$RESULTS/merged"
[[ $CHECK_ONLY -eq 0 ]] && mkdir -p "$MERGED"

printf "%-18s %8s %10s %8s %8s %8s %s\n" SWEEP SHARDS RECORDS KEPT DUPES BROKEN STATUS
printf "%s\n" "-------------------------------------------------------------------------------------"

# Does the dedup/filter pass once per sweep: reads every shard, groups by cell key, keeps the newest record
# per group by timestamp_utc, drops unparseable (truncated) lines, and reports what it did on stderr as
# "__STATS__ total kept dupes broken mixed_commit_cells" plus up to 5 "__WARN__" lines for mixed-commit cells.
# Deduped records go to stdout, so this same call both produces the merge output and the summary line.
dedup_py() {
python3 - "$@" <<'PY'
import json, sys

CELL_KEYS = ("algorithm", "env", "seed", "num_users", "concentration")
kept, commits_seen = {}, {}
broken = total = 0

for path in sys.argv[1:]:
    try:
        handle = open(path)
    except OSError:
        continue
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                broken += 1
                continue
            key = tuple(record.get(k) for k in CELL_KEYS)
            commits_seen.setdefault(key, set()).add(record.get("git_commit"))
            incumbent = kept.get(key)
            if incumbent is None or record.get("timestamp_utc", "") >= incumbent.get("timestamp_utc", ""):
                kept[key] = record

dupes = total - broken - len(kept)
mixed = [k for k, c in commits_seen.items() if len(c) > 1]

for record in kept.values():
    print(json.dumps(record))

print(f"__STATS__ {total} {len(kept)} {dupes} {broken} {len(mixed)}", file=sys.stderr)
for key in mixed[:5]:
    cell = dict(zip(CELL_KEYS, key))
    print(f"__WARN__ mixed git_commit for cell {cell}: {sorted(c or 'None' for c in commits_seen[key])}", file=sys.stderr)
if len(mixed) > 5:
    print(f"__WARN__ ... and {len(mixed) - 5} more cell(s) with mixed git_commit", file=sys.stderr)
PY
}

shopt -s nullglob
for dir in "$RESULTS"/*/; do
    name=$(basename "$dir")
    [[ $name == merged ]] && continue
    wanted "$name" || continue
    shards=("$dir"shard_*.jsonl)
    [[ ${#shards[@]} -eq 0 ]] && continue

    if [[ $CHECK_ONLY -eq 1 ]]; then
        stderr_out=$(dedup_py "${shards[@]}" 2>&1 >/dev/null)
    else
        stderr_out=$(dedup_py "${shards[@]}" 2>&1 >"$MERGED/$name.jsonl")
    fi

    read -r total kept dupes broken mixed <<<"$(echo "$stderr_out" | grep '^__STATS__' | sed 's/__STATS__//')"

    status="ok"
    [[ ${broken:-0} -gt 0 ]] && status="${broken} truncated tail(s) dropped"
    [[ ${dupes:-0} -gt 0 ]] && status="$status; ${dupes} duplicate record(s) collapsed"
    [[ ${mixed:-0} -gt 0 ]] && status="$status; ${mixed} cell(s) with mixed git_commit -- see warnings below"

    printf "%-18s %8s %10s %8s %8s %8s %s\n" "$name" "${#shards[@]}" "${total:-0}" "${kept:-0}" "${dupes:-0}" "${broken:-0}" "$status"
    echo "$stderr_out" | grep '^__WARN__' | sed 's/^__WARN__/    warning:/' || true
done

echo ""
if [[ $CHECK_ONLY -eq 1 ]]; then
    echo "Check only; nothing written."
else
    echo "Merged (deduplicated) into $MERGED/. Next:"
    echo "  python experiments/musols_benchmark/analyze.py $MERGED/*.jsonl"
    echo "  python experiments/musols_benchmark/make_paper_assets.py $MERGED/*.jsonl --out paper_assets"
fi
