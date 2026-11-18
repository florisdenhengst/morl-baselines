#!/usr/bin/env bash
# Merges the per-shard result files into one file per sweep, then reports what arrived.
#
#   ./collect.sh                                     # merge every sweep into results/merged/
#   ./collect.sh --check                             # report completeness without writing anything
#   ./collect.sh resource-gathering                   # merge only the named sweep(s)
#   ./collect.sh --check resource-gathering dst       # check a subset without writing anything
#   ./collect.sh --exclude-commit abc1234 synthetic   # drop every record from a known-bad commit, then merge
#   ./collect.sh --exclude-commit abc1234,def5678     # ... or from several, as one comma/space-separated list
#   ./collect.sh --only-commit abc1234                # keep *only* records from the listed commit(s)
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
#
# The commit filters resolve that judgment call explicitly. Both run *before* the keep-latest step, so a
# filtered-out commit that happens to be the newest one in a mixed-commit group is never preferred just for
# being newest.
#
#   --exclude-commit C   drop every record whose git_commit matches C (blocklist; EXCL column)
#   --only-commit C      drop every record whose git_commit does *not* match C (allowlist; UNSEL column)
#
# Both flags take a single commit, a comma- or space-separated list, and may be repeated; all occurrences
# accumulate, and --flag=value works as well as --flag value. --only-commit is applied first and
# --exclude-commit second, so if a commit matches both it is dropped (and counted under EXCL). A record with
# no git_commit field at all can never match, so --only-commit drops it.
#
# A commit is matched either exactly or as a leading abbreviation of at least 7 characters, so the short
# hashes that git prints work against the full hashes stored in the records. Tokens that matched nothing in
# any merged sweep are reported at the end -- worth reading, since a typo in --only-commit silently discards
# every record rather than too few -- and an abbreviation that matched several distinct commits is warned
# about per sweep.
set -euo pipefail

CHECK_ONLY=0
EXCLUDE_COMMITS=()
ONLY_COMMITS=()

# Splits "abc,def ghi" into SPLIT_COMMITS=(abc def ghi). A global rather than a nameref parameter so this
# still runs under the bash 3.2 that ships with macOS.
split_commits() {
    SPLIT_COMMITS=()
    local token IFS=$', \t\n'
    for token in $1; do
        [[ -n $token ]] && SPLIT_COMMITS+=("$token")
    done
}

while [[ ${1:-} == --* ]]; do
    flag=${1%%=*}
    if [[ $1 == *=* ]]; then value=${1#*=}; consume=1; else value=${2:-}; consume=2; fi
    case "$flag" in
        --check) CHECK_ONLY=1; shift ;;
        --exclude-commit|--only-commit)
            [[ -n $value ]] || { echo "$flag needs a commit (or a comma-separated list of them)" >&2; exit 1; }
            split_commits "$value"
            if [[ ${#SPLIT_COMMITS[@]} -gt 0 ]]; then
                if [[ $flag == --exclude-commit ]]; then
                    EXCLUDE_COMMITS+=("${SPLIT_COMMITS[@]}")
                else
                    ONLY_COMMITS+=("${SPLIT_COMMITS[@]}")
                fi
            fi
            shift "$consume" ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done
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

# Passed to the python helper via env vars rather than argv, so they never collide with shard file paths.
export MUSOLS_EXCLUDE_COMMITS="$(IFS=,; echo "${EXCLUDE_COMMITS[*]:-}")"
export MUSOLS_ONLY_COMMITS="$(IFS=,; echo "${ONLY_COMMITS[*]:-}")"

# Per-token match counts from every sweep, summed once at the end to report tokens that matched nothing
# anywhere. Collected across sweeps because a commit legitimately appears in only one of them, so warning
# per sweep would fire for all the others.
TOKEN_LOG=$(mktemp -t musols_collect_tokens)
trap 'rm -f "$TOKEN_LOG"' EXIT

printf "%-18s %8s %10s %8s %8s %8s %8s %8s %s\n" SWEEP SHARDS RECORDS KEPT DUPES BROKEN EXCL UNSEL STATUS
printf "%s\n" "--------------------------------------------------------------------------------------------------------"

# Does the dedup/filter pass once per sweep: reads every shard, applies the commit filters, groups what
# survives by cell key, keeps the newest record per group by timestamp_utc, drops unparseable (truncated)
# lines, and reports what it did on stderr as
# "__STATS__ total kept dupes broken mixed_commit_cells excluded unselected", one
# "__TOKENS__ <flag> <token> <distinct commits matched>" line per filter token, and up to 5 "__WARN__" lines
# for mixed-commit cells. Deduped records go to stdout, so this same call both produces the merge output and
# the summary line.
dedup_py() {
python3 - "$@" <<'PY'
import json, os, sys

CELL_KEYS = ("algorithm", "env", "seed", "num_users", "concentration")
MIN_PREFIX = 7   # shortest abbreviation we prefix-match; anything shorter has to be an exact hash


def tokens(var):
    return [t for t in os.environ.get(var, "").split(",") if t]


only_tokens = tokens("MUSOLS_ONLY_COMMITS")
exclude_tokens = tokens("MUSOLS_EXCLUDE_COMMITS")
# token -> set of full commits it matched, which tells us both whether it matched anything at all and
# whether an abbreviation was ambiguous.
hits = {t: set() for t in only_tokens + exclude_tokens}


def matches(commit, token_list):
    if not commit:
        return False
    matched = False
    for t in token_list:
        if commit == t or (len(t) >= MIN_PREFIX and commit.startswith(t)):
            hits[t].add(commit)
            matched = True
    return matched


kept, commits_seen = {}, {}
broken = total = excluded = unselected = 0

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
            commit = record.get("git_commit")
            if only_tokens and not matches(commit, only_tokens):
                unselected += 1
                continue
            if exclude_tokens and matches(commit, exclude_tokens):
                excluded += 1
                continue
            key = tuple(record.get(k) for k in CELL_KEYS)
            commits_seen.setdefault(key, set()).add(commit)
            incumbent = kept.get(key)
            if incumbent is None or record.get("timestamp_utc", "") >= incumbent.get("timestamp_utc", ""):
                kept[key] = record

dupes = total - broken - excluded - unselected - len(kept)
mixed = [k for k, c in commits_seen.items() if len(c) > 1]

for record in kept.values():
    print(json.dumps(record))

print(f"__STATS__ {total} {len(kept)} {dupes} {broken} {len(mixed)} {excluded} {unselected}", file=sys.stderr)
for flag, toks in (("--only-commit", only_tokens), ("--exclude-commit", exclude_tokens)):
    for t in toks:
        print(f"__TOKENS__ {flag} {t} {len(hits[t])}", file=sys.stderr)
for t in sorted(set(only_tokens) | set(exclude_tokens)):
    if len(hits[t]) > 1:
        print(f"__WARN__ commit abbreviation {t} matched {len(hits[t])} different commits: {sorted(hits[t])}", file=sys.stderr)
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

    read -r total kept dupes broken mixed excluded unselected <<<"$(echo "$stderr_out" | grep '^__STATS__' | sed 's/__STATS__//')"
    echo "$stderr_out" | grep '^__TOKENS__' >>"$TOKEN_LOG" || true

    # Built up as a list so an empty one means "ok" -- appending to a literal "ok" would otherwise produce
    # readings like "ok; NOTHING KEPT".
    status=""
    note() { status="${status:+$status; }$1"; }
    [[ ${broken:-0} -gt 0 ]] && note "${broken} truncated tail(s) dropped"
    [[ ${dupes:-0} -gt 0 ]] && note "${dupes} duplicate record(s) collapsed"
    [[ ${unselected:-0} -gt 0 ]] && note "${unselected} record(s) not on the --only-commit list"
    [[ ${excluded:-0} -gt 0 ]] && note "${excluded} record(s) excluded by --exclude-commit"
    [[ ${mixed:-0} -gt 0 ]] && note "${mixed} cell(s) with mixed git_commit -- see warnings below"
    [[ ${kept:-0} -eq 0 ]] && note "NOTHING KEPT -- no record survived the filters"
    status=${status:-ok}

    printf "%-18s %8s %10s %8s %8s %8s %8s %8s %s\n" "$name" "${#shards[@]}" "${total:-0}" "${kept:-0}" "${dupes:-0}" "${broken:-0}" "${excluded:-0}" "${unselected:-0}" "$status"
    echo "$stderr_out" | grep '^__WARN__' | sed 's/^__WARN__/    warning:/' || true
done

# A token that matched nothing anywhere is almost always a typo or a wrong-length abbreviation.
awk '{ hits[$2 SUBSEP $3] += $4 }
     END { for (k in hits) if (hits[k] == 0) { split(k, p, SUBSEP);
               print "warning: " p[1] " " p[2] " matched no record in any merged sweep" } }' "$TOKEN_LOG"

echo ""
if [[ $CHECK_ONLY -eq 1 ]]; then
    echo "Check only; nothing written."
else
    echo "Merged (deduplicated) into $MERGED/. Next:"
    echo "  python experiments/musols_benchmark/analyze.py $MERGED/*.jsonl"
    echo "  python experiments/musols_benchmark/make_paper_assets.py $MERGED/*.jsonl --out paper_assets"
fi
