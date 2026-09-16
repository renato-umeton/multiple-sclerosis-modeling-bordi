# Pull request

## What this changes

Say what the change does and why. Link the issue it closes. If the change takes
a deliberate deviation from the article or from the obvious implementation,
record it in `docs/decisions.md` and say so here.

## How it was checked

Paste the commands you ran and their last lines.

## Checklist

- [ ] Tests added for the new behaviour, and they fail without the change.
- [ ] `uv run pytest` is green.
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean.
- [ ] `uv run mypy` is clean under strict mode.
- [ ] `docs/CHANGELOG.md` has an entry under the unreleased heading.
- [ ] No module computes with a number from the article outside
      `src/msrelapse/_params.py`; the code imports `PAPER` instead.
- [ ] Public functions and classes carry a numpy style docstring.
- [ ] Notebooks that were touched have been rebuilt with
      `notebooks/build_notebooks.py` and execute cleanly under
      `uv run --group notebooks pytest -m notebook`; their outputs stay
      stripped, which `nbstripout` enforces.
- [ ] Documentation updated where the change is visible to a reader.
