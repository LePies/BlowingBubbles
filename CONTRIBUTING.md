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

## Branching model

The branch structure mirrors [dtudk/nest-soc](https://github.com/dtudk/nest-soc):

- `main` is the default branch and only receives releases (`dev` → `main`) or
  hotfixes (`bug-fix/*` → `main`). Which source branches may target `main` is
  enforced by the `Source branch` status check
  (`.github/workflows/main-source-branch.yml`), required by the `main` ruleset
  in `.github/rulesets/main.json`.
- `dev` is the integration branch. Cut `feature/*`, `fix/*`, `chore/*` and
  Dependabot branches from `dev` and open PRs into `dev`.
- For PRs into `main`, pick the main template by adding `?template=main.md` to
  the compare URL (`.github/PULL_REQUEST_TEMPLATE/main.md`).

## Repository rulesets

The rulesets in [.github/rulesets/](.github/rulesets) mirror dtudk/nest-soc and
are applied through the GitHub API:

```bash
gh api -X POST repos/LePies/BlowingBubbles/rulesets --input .github/rulesets/dev.json
gh api -X POST repos/LePies/BlowingBubbles/rulesets --input .github/rulesets/main.json
```

Repository rulesets on a private repository require GitHub Pro/Team (or a
public repository). On the free plan these files are the source of truth; the
`Source branch` check still runs on every PR into `main`, and CI must pass on
every PR.

## Pull requests

- Keep changes focused; one logical change per PR.
- PRs target `dev`; fill in the PR template checklist (lint, format, type
  check, deptry, tests).
- CI (`.github/workflows/ci.yml`) must pass on the PR; docs changes are built by
  `.github/workflows/docs.yml`.
- Releases are cut by pushing a `v*.*.*` tag on `main`;
  `.github/workflows/release.yml` builds the distributions and creates the GitHub
  release. Update [CHANGELOG.md](CHANGELOG.md) before tagging.
