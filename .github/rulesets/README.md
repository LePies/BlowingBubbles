# Rulesets

Branch rulesets mirroring [dtudk/nest-soc](https://github.com/dtudk/nest-soc).
Apply them with the GitHub API:

```bash
gh api -X POST repos/LePies/BlowingBubbles/rulesets --input .github/rulesets/dev.json
gh api -X POST repos/LePies/BlowingBubbles/rulesets --input .github/rulesets/main.json
```

- `dev.json` protects `dev`: no deletion, no force-push, and changes only via
  PR with 1 approval and all review threads resolved.
- `main.json` protects the default branch (`main`): no deletion, no force-push,
  merge commits only, 2 approvals, and the required `Source branch` status
  check from `.github/workflows/main-source-branch.yml`, so only `dev` and
  `bug-fix/*` can be merged into `main`.

Repository rulesets on a private repository require GitHub Pro/Team (or a
public repository). Until then, treat these files as the source of truth.
