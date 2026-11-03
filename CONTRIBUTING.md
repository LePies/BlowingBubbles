# Contributing

BlowingBubbles uses [uv](https://docs.astral.sh/uv/) as its package manager. All
commands below assume you are at the repository root.

## Setup

```bash
uv sync                 # installs the default groups (dev + docs)
uv sync --group ui      # additionally install the Streamlit UI dependencies
```

The required Python version is pinned in [.python-version](.python-version);
`uv` picks it up automatically.

Optionally install [pre-commit](https://pre-commit.com/) hooks to run the checks
locally on every commit:

```bash
uv run pre-commit install   # or: pipx run pre-commit install
```

## Development workflow

Run the same checks CI runs:

```bash
uv run ruff check          # lint
uv run ruff format         # auto-format (CI checks with --check)
uv run ty check            # type check (see ty.toml for the checked paths)
uv run deptry .            # dependency hygiene
uv run pytest -q           # tests
uv run pytest -q --cov=blowing_bubbles --cov-report=term   # tests + coverage
```

Ruff and ty are configured in [ruff.toml](ruff.toml) and [ty.toml](ty.toml);
pytest, coverage and deptry are configured in [pyproject.toml](pyproject.toml).
CI uploads the coverage report to Codecov. The UI tests
(`src/tests/test_ui_*.py`) need the `ui` group and run in a separate CI job:

```bash
uv sync --no-default-groups --group dev --group ui
uv run pytest -q src/tests/test_ui_smoke.py src/tests/test_ui_batch.py src/tests/test_ui_desktop.py
```

## Pull requests

- Keep changes focused; one logical change per PR.
- Fill in the PR template checklist (lint, format, type check, deptry, tests).
- CI (`.github/workflows/ci.yml`) must pass on the PR; docs changes are built by
  `.github/workflows/docs.yml`.
- Releases are cut by pushing a `v*.*.*` tag; `.github/workflows/release.yml`
  builds the distributions and creates the GitHub release. Update
  [CHANGELOG.md](CHANGELOG.md) before tagging.
