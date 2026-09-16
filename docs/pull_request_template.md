# Pull request

## What this changes

Say what the change does and why. Link the issue it closes, and the stage of
`docs/IMPLEMENTATION_PLAN.md` it belongs to when there is one.

## How it was checked

Paste the commands you ran and their last lines.

## Checklist

- [ ] Tests added for the new behaviour, and they fail without the change.
- [ ] `uv run pytest` is green.
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean.
- [ ] `uv run mypy` is clean under strict mode.
- [ ] `docs/CHANGELOG.md` has an entry under the unreleased heading.
- [ ] No number from the article is written outside `src/msrelapse/_params.py`;
      the code imports `PAPER` instead.
- [ ] Public functions and classes carry a numpy style docstring.
- [ ] Notebooks that were touched have been rebuilt and their outputs are
      current.
- [ ] Documentation updated where the change is visible to a reader.
