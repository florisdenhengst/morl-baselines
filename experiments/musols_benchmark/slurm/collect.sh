#!/usr/bin/env bash
# Merges the per-shard result files into one file per sweep, then reports what arrived.
#
#   ./collect.sh              # merge into results/merged/
#   ./collect.sh --check      # report completeness without writing anything
#
# Each array task writes its own file precisely so that concurrent tasks never interleave writes: a record
# carrying an anytime trajectory is far larger than the atomic-append limit, so appending from many jobs to a
# shared file would corrupt it. Merging is therefore a separate, deliberate step.
set -euo pipefail

CHECK_ONLY=0
[[ ${1:-} == "--check" ]] && CHECK_ONLY=1

cd "$(dirname "$0")"
REPO_ROOT=${MUSOLS_REPO:-$(cd ../../.. && pwd)}
RESULTS="$REPO_ROOT/results"
MERGED="$RESULTS/merged"
[[ $CHECK_ONLY -eq 0 ]] && mkdir -p "$MERGED"

printf "%-18s %8s %10s %12s %s\n" SWEEP SHARDS RECORDS "EMPTY" STATUS
printf "%s\n" "---------------------------------------------------------------------"

shopt -s nullglob
for dir in "$RESULTS"/*/; do
    name=$(basename "$dir")
    [[ $name == merged ]] && continue
    shards=("$dir"shard_*.jsonl)
    [[ ${#shards[@]} -eq 0 ]] && continue

    empty=0
    for f in "${shards[@]}"; do [[ -s $f ]] || empty=$((empty + 1)); done
    records=$(cat "${shards[@]}" 2>/dev/null | wc -l | tr -d ' ')

    # A truncated final line means a task was killed mid-write; the record is unusable and would break any
    # downstream json.loads, so it is worth knowing before analysis rather than during it.
    broken=0
    for f in "${shards[@]}"; do
        [[ -s $f ]] && { tail -c1 "$f" | read -r _ || true; tail -n1 "$f" | python -c "import json,sys; json.loads(sys.stdin.read() or '{}')" 2>/dev/null || broken=$((broken + 1)); }
    done

    status="ok"
    [[ $empty -gt 0 ]] && status="$empty empty shard(s)"
    [[ $broken -gt 0 ]] && status="$status; $broken truncated tail(s)"

    printf "%-18s %8s %10s %12s %s\n" "$name" "${#shards[@]}" "$records" "$empty" "$status"

    if [[ $CHECK_ONLY -eq 0 ]]; then
        # Drop any truncated trailing line rather than propagating a record that cannot be parsed.
        for f in "${shards[@]}"; do
            python - "$f" <<'PY'
import json, sys
path = sys.argv[1]
good = []
with open(path) as handle:
    for line in handle:
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue
        good.append(line)
sys.stdout.writelines(good)
PY
        done > "$MERGED/$name.jsonl"
    fi
done

echo ""
if [[ $CHECK_ONLY -eq 1 ]]; then
    echo "Check only; nothing written."
else
    echo "Merged into $MERGED/. Next:"
    echo "  python experiments/musols_benchmark/analyze.py $MERGED/*.jsonl"
    echo "  python experiments/musols_benchmark/make_paper_assets.py $MERGED/*.jsonl --out paper_assets"
fi
