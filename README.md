<div align="center">

# FinMAS: Financial Analysis using LLM Multi-Agent Systems

<h3>

[Documentation](https://kevorksulahian.github.io/agentic-llm-for-better-results/) | [Example Outputs](https://kevorksulahian.github.io/agentic-llm-for-better-results/examples_index/)

</h3>

[![CrewAI](https://img.shields.io/badge/CrewAI-red)](https://docs.crewai.com/introduction)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-black)](https://docs.llamaindex.ai/en/stable/)
[![OpenAI](https://img.shields.io/badge/OpenAI-green?logo=openai&logoColor=white)](https://platform.openai.com/docs/models)
[![Groq](https://img.shields.io/badge/Groq-red)](https://console.groq.com/docs/overview)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/models?other=embeddings)
[![Panel Hero](https://img.shields.io/badge/Panel-Hero)](https://panel.holoviz.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

This repo contains the code for [WQU](https://www.wqu.edu/mscfe) Capstone project where
we investigate the use of LLM multi-agent systems for solving tasks in the financial domain.
We use the [CrewAI](https://docs.crewai.com/introduction) framework to orchestrate the agents,
and the [LlamaIndex](https://docs.llamaindex.ai/en/stable/) framework to creating vector store
index from unstructured text data like news and SEC filings.

[4 crews](https://kevorksulahian.github.io/agentic-llm-for-better-results/crews/) have been
created that have different focus, with different data sources.
A final [combined crew](https://kevorksulahian.github.io/agentic-llm-for-better-results/crews/combined/) is created
that combines data from news, SEC filings and market data to provide a final stock analysis that
includes a recommendation.

The following screenshots illustrate a output from the combined crew and the main dashboard.

### Combined analysis

![](docs/assets/screenshots/finmas_combined_analysis.png)

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

We recommend using the [uv package manager](https://github.com/astral-sh/uv) to install the dependencies.

From the root of the project run the following command to install the
latest dependencies without the development dependencies:

```shell
uv sync --upgrade --no-dev
```

If you want to use standard pip instead, use the following:

```shell
python -m venv .venv
source .venv/bin/activate  # macOS or Linux
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

3. Set up `.env` file with necessary API keys.

## 2. Running the app

Activate the virtual environment and start the server by using `panel`:

```shell
source .venv/bin/activate  # macOS or Linux
.venv\Scripts\activate  # Windows
panel serve finmas/panel/app.py --show
```

If you want to start the app with a specific ticker like `META`:

```shell
panel serve finmas/panel/app.py --show --args --args META
```

We use [Alpha Vantage](https://www.alphavantage.co/) to get fundamental data (income statements).\
You can create your `.env` file by copying the `.env.template` file in the repo.
Set the following API keys in the `.env` file in the repo folder:

- [ALPHAVANTAGE_API_KEY](https://www.alphavantage.co/) for fundamental data.
- [ALPACA_API_KEY](https://docs.alpaca.markets/docs/historical-news-data) and `ALPACA_API_SECRET` for access to Benzinga Historical News API.
- [GROQ_API_KEY](https://console.groq.com/playground) for access to running Groq models.
- [OPENAI_API_KEY](https://platform.openai.com/settings/organization/api-keys) for accessing OpenAI models `gpt-4o` and `gpt-4o-mini`.
- [HF_TOKEN](https://huggingface.co/settings/tokens) for access to HuggingFace embedding models.

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
