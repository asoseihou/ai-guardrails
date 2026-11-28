# Contributing

Thanks for helping. This is a small project with strict gates, so the checklist below is short but enforced by CI.

## Set up

```bash
make install     # Python 3.12+; creates .venv from the hash-pinned dev lock, CPU PyTorch, editable package
make demo        # offline end-to-end run: synth, train, evaluate, reports
```

## Before opening a pull request

```bash
make check       # ruff lint + format check, mypy --strict, tests + 85% core coverage gate, bandit, pip-audit
```

- Keep the layering (`tests/test_architecture.py` and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)): `domain` and `guardrails` stay pure; interfaces hold no business rules.
- Read configuration through `docai.settings.Settings`, never `os.environ` in a module.
- Type hints everywhere (mypy strict), docstrings on public functions, functions under ~60 lines and complexity 10.
- Code, comments and docs in English. Commit messages in the imperative mood, one logical change per commit.

## Changing a guardrail

A guardrail change must not break a correct value. Before proposing one:

1. Add or update a test with the failure it targets.
2. Run `python -m docai replay --a <before> --b <after>` and include the blast-radius summary in the PR. **Regressed must be 0.**
3. If the change targets a new error mode, add it to the simulator (`ErrorRates` in `extract/providers/simulated.py`) so the report measures it.
4. Run `make demo` and commit the regenerated `reports/` and README table.

The policy is in [ADR 0002](docs/adr/0002-never-invent.md): correct only with proof, otherwise null and send to review.

## Dependencies

Add the dependency to `pyproject.toml` with bounds, then regenerate the locks with `make lock` (needs [uv](https://github.com/astral-sh/uv)) and commit both lock files. CI installs with `--require-hashes`.

## Data

Use only synthetic data. Never commit real documents, real company identifiers, API keys, `.venv`, `data/`, `var/` or model weights.

## Security issues

Report them privately, as described in [SECURITY.md](SECURITY.md).
