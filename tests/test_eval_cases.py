import json
from pathlib import Path

from evals.eval_lib import evaluate_text, summarize


def test_memory_cases_have_at_least_twenty_turns():
    path = Path(__file__).parents[1] / "evals" / "cases" / "memory_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))["turns"]
    assert len(cases) >= 20
    assert [case["turn"] for case in cases] == list(range(1, len(cases) + 1))


def test_eval_scorer_checks_required_and_forbidden_text():
    result = evaluate_text(1, "Atlas runs on Linux", ["Atlas", "Linux"], ["password"])
    assert result.passed
    failed = evaluate_text(2, "password is secret", ["PostgreSQL"], ["password"])
    assert not failed.passed
    assert failed.missing == ("PostgreSQL",)
    assert failed.forbidden == ("password",)
    assert summarize([result, failed])["accuracy"] == 0.5
