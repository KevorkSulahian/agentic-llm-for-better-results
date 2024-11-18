# FinMAS Crews

In the FinMAS app we have configured 4 different crews that can be used to analyze the
data using the LLM agents. Click on the individual crew to see the details of how each
crew is defined in terms of agents and tasks.

To see more general information about the crews, [click here](../components/crews.md).

The crews are defined in the [crews](https://github.com/KevorkSulahian/agentic-llm-for-better-results/tree/main/finmas/crews)
subpackage in the code repository.

| Crew                                | Description                                                                  |
| ----------------------------------- | ---------------------------------------------------------------------------- |
| [Market Data](market_data.md)       | Technical and fundamental analysis summary                                   |
| [News](news.md)                     | Summary and sentiment analysis of Benzinga News                              |
| [SEC MD&A and Risk Factors](sec.md) | Analyze MD&A and Risk Factors sections from a single SEC 10-K or 10-Q filing |
| [Combined](combined.md)             | Combine all the data sources and provide a final stock recommendation        |
