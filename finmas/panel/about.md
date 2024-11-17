## About FinMAS - Financial Multi-agent System

This app uses LLM agents organized in a multi-agent system to analyze financial data
and perform financial tasks. The app is developed during the final capstone project of the
[WorldQuant MSc in Financial Engineering](https://www.wqu.edu/mscfe).

It is meant as a practical and educational app that demonstrates the state-of-the-art of LLM models applied
to tasks in the financial domain, and with an extra focus on open source models and packages.

Please visit [GitHub repo](https://github.com/KevorkSulahian/agentic-llm-for-better-results) and our [docs](https://kevorksulahian.github.io/agentic-llm-for-better-results/) for further information.

### Data Sources

This is a list of the data sources that have been integrated in this app.
Those data sources that require an API Key are marked in the table.
Then it is necessary to set those as a environment variable which can
be done by editing the `.env` file in the root folder.

| API Key                           | Name          | Type             | Description                             |
| --------------------------------- | ------------- | ---------------- | --------------------------------------- |
|                                   | Yahoo Finance | Price data       |                                         |
|                                   | SEC / Edgar   | Filings          |                                         |
| <i class="fa-solid fa-check"></i> | Benzinga      | News             | Requires registration of Alpaca account |
| <i class="fa-solid fa-check"></i> | Alpha Vantage | Fundamental data | Limited to 25 calls per day.            |

### Main Python packages

Here we list some of the main packages that make this app possible:

| Name                                                 | Description                                              |
| ---------------------------------------------------- | -------------------------------------------------------- |
| [crewai](https://docs.crewai.com/introduction)       | Multi-agent orchestration framework                      |
| [llama-index](https://docs.llamaindex.ai/en/stable/) | Data framework to help transform data to be used by LLMs |
| [panel](https://panel.holoviz.org/)                  | Web app framework focused on data science applications   |
| [groq](https://groq.com/)                            | Run hosted open source LLMs in the cloud                 |
| [openai](https://github.com/openai/openai-python)    | Access to gpt-4o and gpt-4o-mini                         |
