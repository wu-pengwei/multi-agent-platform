# Long-horizon memory evaluation

`cases/memory_cases.json` contains 24 ordered turns. The early turns seed
project facts and user preferences; later turns query those facts after enough
intervening context to exceed the fixed recent-history window.

The automatic scorer is intentionally deterministic and framework-independent:
it checks required substrings (`expected_in`) and forbidden substrings
(`expected_not_in`) in one response per turn.

## Running

Export the system under test's responses as JSONL:

```json
{"turn": 20, "text": "Atlas 运行在 Linux 环境。"}
```

Then run:

```bash
python -m evals.run_memory_eval --input responses.jsonl
```

The command prints one PASS/FAIL line per turn, a JSON summary, and exits with
status 1 if any case fails. Run the same cases with semantic recall disabled
and enabled to compare baseline accuracy.
