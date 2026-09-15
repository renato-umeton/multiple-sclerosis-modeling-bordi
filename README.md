# Multiple sclerosis modeling

## Setup

Requires [uv](https://docs.astral.sh/uv/). Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repo and install the environment:

```bash
git clone git@github.com:renato-umeton/multiple-sclerosiss-modeling-bordi.git
cd multiple-sclerosiss-modeling-bordi
uv sync
```

`uv sync` downloads Python 3.14 if needed, creates `.venv`, and installs the
locked dependencies from `uv.lock`.

## Daily use

```bash
uv run python path/to/script.py   # run inside the project environment
uv add <package>           # add a dependency and update uv.lock
uv add --dev <package>     # add a dev-only dependency
uv sync                    # re-sync after pulling changes to uv.lock
```
