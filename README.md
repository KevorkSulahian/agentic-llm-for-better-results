# FinMAS - Financial Analysis by Multi-Agent System

## [Documentation](https://kevorksulahian.github.io/agentic-llm-for-better-results/)

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Panel Hero](https://img.shields.io/badge/Panel-Hero)](https://panel.holoviz.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repo contains the code for WQU Capstone project where we investigate the use of LLM multi-agent systems for solving tasks
in the financial domain. The main focus will be on sentiment analysis, while also maintaining a broader look on how such multi-agent
systems may perform on other financial tasks as well.

The following screenshots illustrate a news analysis crew output and the main dashboard.

### News analysis

![](docs/assets/screenshots/finmas_news_analysis.png)

### Main dashboard

![](docs/assets/screenshots/finmas_main_dashboard.png)

## Web app architecture

The following diagram shows how the different components of the web app are connected together.

![](docs/assets/finmas_architecture.png)

## Getting started

## 1. Installation

To install the app do the following:

1. Clone the repo

```shell
git clone https://github.com/KevorkSulahian/agentic-llm-for-better-results.git
cd agentic-llm-for-better-results
```

2. Create a virtual environment and install the dependencies into the environment.

Using `venv` and `pip`

```shell
python -m venv .venv
source .venv/bin/activate  # macOS or Linux
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Using uv

```shell
uv sync
```

3. Set up `.env` file with necessary API keys.

## 2. Running the app

Activate the virtual environment and start the server by using `panel`:

```shell
source .venv/bin/activate  # macOS or Linux
.venv\Scripts\activate  # Windows
panel serve finmas/panel/app.py --show
```

We use [Alpha Vantage](https://www.alphavantage.co/) to get fundamental data (income statements).\
You can create your `.env` file by copying the `.env.template` file in the repo.
Set the following API keys in the `.env` file in the repo folder:

- `ALPHAVANTAGE_API_KEY` for fundamental data.
- `ALPACA_API_KEY` and `ALPACA_API_SECRET` for access to Benzinga Historical News API.
- `GROQ_API_KEY` for access to running Groq models.
- `OPENAI_API_KEY` for accessing OpenAI models `gpt-4o` and `gpt-4o-mini`.
- `HF_TOKEN` for access to HuggingFace models.

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
