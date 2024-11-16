## Query Engine Help

The llama-index Query Engine can use the Vector Store Index to add context to queries
that is sent to the LLM. Here you can choose to create a Vector Store Index
from either the News data or a section from the SEC filing. This is the same
query engine that the agents in the crews use to solve their tasks.
Here you can experiment with different queries and see how the LLM
responds to them. The LLM will not "remember" the history of the conversation
so every query is independent.

Some example queries you can try are:

**SEC Management Discussion and Analysis (SEC MDA)**

- Extract the most important statements from the Management's Discussion and
  Analysis section of the provided 10-K filing.

**SEC Risk Factors**

- Extract the most important statements from the Risk Factors section of the
  provided 10-K filing.

**News Data**

- Extract the most important statements from the news articles for the
  ticker (Input the ticker for analysis).