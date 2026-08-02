"""
scripts/uat_prep.py — UAT Question Set Loader & Placeholder Checker

Theresa's questionnaire uses two gap conventions that MUST be filled
before questions can be fed to the LLM:

    "[Something]"  → a business value goes here
                     ("orders from [Customer]" → "orders from Müller GmbH")
    "..."          → a number goes here
                     ("more than ... orders" → "more than 10 orders")

This tool:
  1. Loads the questionnaire (xlsx / csv / txt — one question per row).
  2. Flags every question that still contains an unfilled gap.
  3. Auto-detects language (very simple DE/EN heuristic) and assigns
     sequential IDs (UAT001, UAT002, ...).
  4. Writes a batch-runner-ready CSV (id,question,language,category)
     containing ONLY the ready questions, plus a gap report listing
     what still needs a human decision.

Usage:
    python -m erp42_analytics.scripts.uat_prep \
        --input theresa_questions_en.xlsx theresa_questions_de.xlsx \
        --output scripts/uat_questions.csv

Then fill the gaps listed in the report, re-run, and feed the CSV to:
    python -m erp42_analytics.scripts.batch_test_runner \
        --input scripts/uat_questions.csv --run-id pretest_home
"""

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Gap patterns from Theresa's mail
_BRACKET_GAP = re.compile(r"\[[^\]]*\]")        # [Something]
_ELLIPSIS_GAP = re.compile(r"\.\.\.|…")          # ... or unicode ellipsis

# Cheap language heuristic — enough for tagging, not linguistics
_DE_HINTS = re.compile(
    r"\b(wie|viele|welche|umsatz|kunden|aufträge|auftrag|rechnung|"
    r"artikel|liefer|zeig|gibt|haben|wurden|monat|jahr|letzten)\b|[äöüß]",
    re.IGNORECASE,
)


def detect_language(question: str) -> str:
    return "de" if _DE_HINTS.search(question) else "en"


def find_gaps(question: str) -> list:
    gaps = _BRACKET_GAP.findall(question)
    if _ELLIPSIS_GAP.search(question):
        gaps.append("...")
    return gaps


def _read_rows(path: Path) -> list:
    """Extract raw question strings from xlsx / csv / txt."""
    suffix = path.suffix.lower()
    questions = []

    if suffix in (".xlsx", ".xlsm"):
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                for cell in row:
                    if isinstance(cell, str) and len(cell.strip()) > 12 \
                            and "?" in cell:
                        questions.append(cell.strip())
    elif suffix == ".csv":
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                for cell in row:
                    if len(cell.strip()) > 12 and "?" in cell:
                        questions.append(cell.strip())
    else:  # txt — one question per line
        for line in path.read_text(encoding="utf-8").splitlines():
            if len(line.strip()) > 12:
                questions.append(line.strip())

    return questions


def prepare(
    input_paths: list,
    output_csv: str,
    id_prefix: str = "UAT",
) -> dict:
    all_q = []
    for p in input_paths:
        path = Path(p)
        if not path.exists():
            logger.error("Input not found: %s", p)
            continue
        found = _read_rows(path)
        logger.info("%s: %d question(s) extracted", path.name, len(found))
        all_q.extend(found)

    # De-duplicate while preserving order
    seen = set()
    unique_q = []
    for q in all_q:
        key = q.lower().strip()
        if key not in seen:
            seen.add(key)
            unique_q.append(q)

    ready, gapped = [], []
    for i, q in enumerate(unique_q, 1):
        entry = {
            "id": f"{id_prefix}{i:03d}",
            "question": q,
            "language": detect_language(q),
            "category": "",   # optional — filled manually if desired
        }
        gaps = find_gaps(q)
        if gaps:
            entry["gaps"] = gaps
            gapped.append(entry)
        else:
            ready.append(entry)

    # Write batch-ready CSV (ready questions only)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "question", "language", "category"]
        )
        writer.writeheader()
        writer.writerows(ready)

    # Write gap report next to it
    report_path = out.with_name(out.stem + "_GAPS.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            f"UAT GAP REPORT — {len(gapped)} question(s) still need "
            f"values filled in before testing\n"
            + "=" * 60 + "\n\n"
        )
        for e in gapped:
            f.write(f"{e['id']} [{e['language']}]\n")
            f.write(f"  Q: {e['question']}\n")
            f.write(f"  Fill: {', '.join(e['gaps'])}\n\n")

    summary = {
        "total": len(unique_q),
        "ready": len(ready),
        "needs_filling": len(gapped),
        "output_csv": str(out),
        "gap_report": str(report_path),
        "by_language": {
            "de": sum(1 for e in ready + gapped if e["language"] == "de"),
            "en": sum(1 for e in ready + gapped if e["language"] == "en"),
        },
    }
    logger.info(
        "UAT prep: %d total | %d ready → %s | %d need gap-filling → %s",
        summary["total"], summary["ready"], out.name,
        summary["needs_filling"], report_path.name,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Theresa's UAT questionnaire for batch testing"
    )
    parser.add_argument("--input", nargs="+", required=True,
                        help="questionnaire file(s): xlsx / csv / txt")
    parser.add_argument("--output", default="scripts/uat_questions.csv")
    parser.add_argument("--id-prefix", default="UAT")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = prepare(args.input, args.output, args.id_prefix)

    if summary["needs_filling"]:
        print(
            f"\n⚠  {summary['needs_filling']} question(s) contain "
            f"[Something]/... gaps.\n"
            f"   Open {summary['gap_report']}, fill real values into the "
            f"source file, and re-run this tool."
        )
        sys.exit(1)
    print(f"\n✓ All {summary['ready']} questions ready: "
          f"{summary['output_csv']}")


if __name__ == "__main__":
    main()
