import time
from typing import Optional

import pandas as pd
import typer
import util
from config import DataSetEnum, LlmModelEnum
from crewai import Agent, Crew, Process, Task
from rich import print
from tabulate import tabulate
from typing_extensions import Annotated


def encode_sentiment(sentiment: str) -> int:
    sentiment = sentiment.lower().strip().rstrip(".")
    if sentiment == "negative":
        return 0
    elif sentiment == "neutral":
        return 1
    elif sentiment == "positive":
        return 2
    else:
        raise ValueError(f"Invalid sentiment: {sentiment}")


def main(
    model_name: Annotated[
        LlmModelEnum,
        typer.Argument(help="Model name"),
    ] = LlmModelEnum.llama3_1,
    dataset: Annotated[DataSetEnum, typer.Argument(help="Dataset name")] = DataSetEnum.fpb,
    limit: Annotated[
        Optional[int], typer.Option(help="Number of entries from dataset to analyze")
    ] = None,
    show: Annotated[bool, typer.Option(help="Show confusion matrix")] = False,
):
    """
    Use an LLM as an Agent to run sentiment analysis on a dataset with crewAI
    """
    if dataset == "fpb":
        df = util.get_fpb_data()
        if limit:
            df = df.head(limit)
        records = df.to_dict(orient="records")
    else:
        print("Unsupported dataset")
        raise typer.Exit(1)

    print(f"Running sentiment analysis on {len(records)} records")
    print("Preview of dataset")
    with pd.option_context("display.max_colwidth", 200):
        print(df.head(10))
    typer.confirm("Proceed with sentiment analysis?", abort=True)

    model = f"ollama/{model_name.value}"

    print(f"Running sentiment analysis on dataset: {dataset} using {model}")
    if limit:
        print(f"Limiting to {limit} records")

    # Setup crewAI system
    sa_agent = Agent(
        role="Sentiment Analysis Agent",
        goal="Classify the financial sentiment of the phrase with a single word as positive, negative, or neutral",
        backstory="You are an expert in financial sentiment analysis.",
        verbose=True,
        allow_delegation=False,
        llm=model,
    )

    sa_task = Task(
        description="Analyze the sentiment of this statement extracted from a financial news article. "
        "Provide your answer as either negative, positive or neutral: {phrase}",
        expected_output="A single word indicating the sentiment: positive, negative, or neutral",
        agent=sa_agent,
    )

    crew = Crew(
        agents=[sa_agent],
        tasks=[sa_task],
        verbose=True,
        output_log_file="sa_agent.log",
        memory=False,  # Default option
        process=Process.sequential,  # Default option
    )

    # Perform sentiment analysis
    results = crew.kickoff_for_each(inputs=records)

    outputs = [encode_sentiment(r.raw) for r in results]

    # Output result
    result_df = pd.DataFrame({"label": df["label"], "predicted": outputs})
    result_df["correct"] = result_df["predicted"] == result_df["label"]
    print("Accuracy: ", result_df["correct"].mean())
    print("Preview of dataset with predictions")
    print(tabulate(result_df, headers="keys"))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = f"sa_results_{model_name.value}_{dataset.value}_{timestamp}.csv"
    result_df.to_csv(output, index=False)
    print(f"Results saved to {output}")

    util.output_performance_summary(
        df["label"], result_df["predicted"], model_name.value, dataset.value, show
    )


if __name__ == "__main__":
    typer.run(main)
