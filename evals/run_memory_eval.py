"""Run deterministic memory recall checks against a JSONL history or prompt dump.

Usage:
  python -m evals.run_memory_eval --input responses.jsonl

Each input line is JSON with ``turn`` and ``text``. The runner joins it with
``cases/memory_cases.json`` and exits non-zero when any case fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.eval_lib import evaluate_text, summarize

ROOT = Path(__file__).parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate long-horizon memory recall")
    parser.add_argument("--input", type=Path, required=True, help="JSONL responses: {turn, text}")
    parser.add_argument("--cases", type=Path, default=ROOT / "cases" / "memory_cases.json")
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))["turns"]
    responses = {
        int(row["turn"]): str(row.get("text", ""))
        for row in (json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines())
        if row.get("turn") is not None
    }
    results = [
        evaluate_text(
            case["turn"],
            responses.get(case["turn"], ""),
            case.get("expected_in", []),
            case.get("expected_not_in", []),
        )
        for case in cases
    ]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        details = []
        if result.missing:
            details.append("missing=" + ",".join(result.missing))
        if result.forbidden:
            details.append("forbidden=" + ",".join(result.forbidden))
        print(f"{status} turn={result.turn}" + (" " + " ".join(details) if details else ""))
    summary = summarize(results)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
