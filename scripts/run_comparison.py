"""
scripts/run_comparison.py — Run Comparison Engine (Step 1.11)

Compares two stored test runs question-by-question and answers the
Week-2 question: "did our fixes actually work, and did we break
anything?"

Per-question verdicts:
    FIXED       failed in A, succeeds in B
    REGRESSED   succeeded in A, fails in B     ← the dangerous one
    STILL_FAIL  failed in both (with category change noted)
    STABLE      succeeded in both

Also aggregates failure_stage / error_category counts per run so the
taxonomy trend is visible (e.g. "hallucinated_table: 14 → 3").

CLI:
    python -m erp42_analytics.scripts.run_comparison \
        --baseline baseline_w1 --candidate after_fixes_w2
"""

import argparse
import logging
from collections import Counter

from erp42_analytics.scripts.test_run_repository import TestRunRepository

logger = logging.getLogger(__name__)


def compare_runs(
    baseline_id: str,
    candidate_id: str,
    repo: TestRunRepository = None,
) -> dict:
    repo = repo or TestRunRepository()
    run_a = repo.get_run(baseline_id)
    run_b = repo.get_run(candidate_id)
    if not run_a or not run_b:
        missing = baseline_id if not run_a else candidate_id
        raise ValueError(f"Run not found in repository: '{missing}'")

    res_a = {r["question_id"]: r for r in repo.get_results(baseline_id)}
    res_b = {r["question_id"]: r for r in repo.get_results(candidate_id)}

    common = sorted(set(res_a) & set(res_b))
    only_a = sorted(set(res_a) - set(res_b))
    only_b = sorted(set(res_b) - set(res_a))

    verdicts = {"FIXED": [], "REGRESSED": [], "STILL_FAIL": [], "STABLE": []}

    for qid in common:
        a, b = res_a[qid], res_b[qid]
        ok_a, ok_b = bool(a["success"]), bool(b["success"])
        if not ok_a and ok_b:
            verdicts["FIXED"].append(qid)
        elif ok_a and not ok_b:
            verdicts["REGRESSED"].append({
                "question_id": qid,
                "new_failure_stage": b["failure_stage"],
                "new_error_category": b["error_category"],
                "error": (b["error"] or "")[:120],
            })
        elif not ok_a and not ok_b:
            entry = {"question_id": qid}
            if a["error_category"] != b["error_category"]:
                entry["category_change"] = (
                    f"{a['error_category'] or '?'} → "
                    f"{b['error_category'] or '?'}"
                )
            verdicts["STILL_FAIL"].append(entry)
        else:
            verdicts["STABLE"].append(qid)

    def _taxonomy(results: dict) -> dict:
        failed = [r for r in results.values() if not r["success"]]
        return {
            "by_stage": dict(Counter(
                r["failure_stage"] or "unknown" for r in failed
            )),
            "by_category": dict(Counter(
                r["error_category"] or "unknown" for r in failed
            )),
        }

    return {
        "baseline": {
            "run_id": baseline_id,
            "success_rate": run_a["success_rate"],
            "avg_time_s": run_a["avg_time_s"],
            "taxonomy": _taxonomy(res_a),
        },
        "candidate": {
            "run_id": candidate_id,
            "success_rate": run_b["success_rate"],
            "avg_time_s": run_b["avg_time_s"],
            "taxonomy": _taxonomy(res_b),
        },
        "delta": {
            "success_rate": round(
                run_b["success_rate"] - run_a["success_rate"], 1
            ),
            "avg_time_s": round(
                run_b["avg_time_s"] - run_a["avg_time_s"], 1
            ),
        },
        "verdicts": verdicts,
        "counts": {k: len(v) for k, v in verdicts.items()},
        "coverage": {
            "common": len(common),
            "only_in_baseline": only_a,
            "only_in_candidate": only_b,
        },
    }


def format_report(cmp: dict) -> str:
    """Human-readable comparison report (for the weekly sync)."""
    a, b, d = cmp["baseline"], cmp["candidate"], cmp["delta"]
    lines = [
        "=" * 62,
        "RUN COMPARISON REPORT",
        "=" * 62,
        f"Baseline : {a['run_id']}   "
        f"success {a['success_rate']}%   avg {a['avg_time_s']}s",
        f"Candidate: {b['run_id']}   "
        f"success {b['success_rate']}%   avg {b['avg_time_s']}s",
        f"Delta    : success {d['success_rate']:+.1f}pp   "
        f"time {d['avg_time_s']:+.1f}s",
        "-" * 62,
        f"FIXED      : {cmp['counts']['FIXED']}",
        f"REGRESSED  : {cmp['counts']['REGRESSED']}   ← investigate first",
        f"STILL_FAIL : {cmp['counts']['STILL_FAIL']}",
        f"STABLE     : {cmp['counts']['STABLE']}",
    ]
    if cmp["verdicts"]["REGRESSED"]:
        lines.append("-" * 62)
        lines.append("Regressions:")
        for r in cmp["verdicts"]["REGRESSED"]:
            lines.append(
                f"  {r['question_id']}: "
                f"[{r['new_failure_stage']}/{r['new_error_category']}] "
                f"{r['error']}"
            )
    lines.append("-" * 62)
    lines.append("Failure taxonomy (baseline → candidate):")
    cats = set(a["taxonomy"]["by_category"]) | set(
        b["taxonomy"]["by_category"]
    )
    for c in sorted(cats):
        va = a["taxonomy"]["by_category"].get(c, 0)
        vb = b["taxonomy"]["by_category"].get(c, 0)
        lines.append(f"  {c:<24} {va:>3} → {vb:<3}")
    lines.append("=" * 62)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare two test runs")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(format_report(compare_runs(args.baseline, args.candidate)))


if __name__ == "__main__":
    main()
