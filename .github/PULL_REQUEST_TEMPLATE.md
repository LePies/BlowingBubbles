<!--
PRs go into dev and need 1 approval from someone other than the author.
Only dev and bug-fix/* branches may be merged into main; for those, use the
main template by adding ?template=main.md to the "compare" URL.
-->

## Summary

<!-- What does this PR change and why? -->

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation / tooling / CI

## Verification

<!-- Commands run locally, e.g. `uv run ruff check`, `uv run pytest -q` -->

- [ ] The PR targets `dev`
- [ ] `uv run ruff check` passes
- [ ] `uv run ruff format --check` passes
- [ ] `uv run ty check` passes
- [ ] `uv run deptry .` passes
- [ ] `uv run pytest -q` passes
