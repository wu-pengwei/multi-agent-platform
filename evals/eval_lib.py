"""Framework-neutral assertions for long-horizon memory evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class EvalResult:
    turn: int
    passed: bool
    missing: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()


def evaluate_text(
    turn: int,
    text: str,
    expected_in: Iterable[str] = (),
    expected_not_in: Iterable[str] = (),
) -> EvalResult:
    """Case-insensitive substring scoring for deterministic regression tests."""
    haystack = text.casefold()
    missing = tuple(item for item in expected_in if item.casefold() not in haystack)
    forbidden = tuple(item for item in expected_not_in if item.casefold() in haystack)
    return EvalResult(turn, not missing and not forbidden, missing, forbidden)


def summarize(results: list[EvalResult]) -> dict[str, float | int]:
    passed = sum(result.passed for result in results)
    total = len(results)
    return {"passed": passed, "total": total, "accuracy": passed / total if total else 0.0}
