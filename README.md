# FinMAS - Financial Analysis by Multi-Agent System

[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This repo contains the code for WQU Capstone project where we investigate the use of LLM multi-agent systems for solving tasks
in the financial domain. The main focus will be on sentiment analysis, while also maintaining a broader look on how such multi-agent
systems may perform on other financial tasks as well.

## Usage

To run the app do the following:

1. Clone the repo
2. Install the virtual environment and activate it.
3. Setup `.env` file
4. Start the server by using `panel serve`

We use [Alpha Vantage](https://www.alphavantage.co/) to get fundamental data (income statements).  
Set your `ALPHAVANTAGE_API_KEY` in the `.env` file in the repo folder.  
You can create your `.env` file by copying the `.env.template` file in the repo.

From the repo folder, the following command can start the app:

```bash
panel serve finmas/panel/app.py --show
```

## Development

The project setup is inspired by both [Python for Data Science](https://www.python4data.science/en/latest/productive/index.html) and
the [Learn Scientific Python](https://learn.scientific-python.org/development/guides/style/) project. These projects give guidelines
to how to set up a research project that is reproducible and with good quality.

Commit messages are encouraged to follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

### Virtual environment

To install the virtual environment, we use the extremely fast [uv project manager](https://github.com/astral-sh/uv).
Install `uv` using [the standalone installer](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) is recommended.
To create or sync a virtual environment the following command can be used in the project folder:

```bash
uv sync
```

To exclude the development dependencies (like `pre-commit`) append the `--no-dev` flag to the command:

```bash
uv sync --no-dev
```

To add or remove packages, simply use `uv add <package>` or `uv remove <package>`.

Activate the virtual environment with the following command:

```bash
source .venv/bin/activate  # macOS or Linux
.venv\Scripts\activate  # Windows
```

### Styling and pre-commit

To maintain the code quality when committing to the repo we use [pre-commit](https://pre-commit.com/) with
ruff, type checking for script files and formatting of pyproject.toml file. This ensures that these
code quality tools are run before any commit.

The configuration is stored in `.pre-commit-config.yaml`, and to set up the git hook scripts simply run
the following in the virtual environment:

```bash
pre-commit install
```

The pre-commits can be run on all files before committing by this command:

```bash
pre-commit run --all-files
```
