from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from crewai import Agent, Crew, Process, Task
from datasets import load_dataset
from rich import print
from tabulate import tabulate
from typing_extensions import Annotated

OLLAMA_MODELS = ["llama3.1", "mistral-nemo"]
DATASETS = ["fpb"]

FPB_CONFIGURATION = "sentences_50agree"


def get_fpb_data() -> pd.DataFrame:
    if Path.is_file(Path(f"fpb_{FPB_CONFIGURATION}.csv")):
        print("Using cached FPB dataset")
        df = pd.read_csv(f"fpb_{FPB_CONFIGURATION}.csv")
    else:
        print("Downloading FPB dataset")
        dataset = load_dataset(
            path="takala/financial_phrasebank", name=FPB_CONFIGURATION, trust_remote_code=True
        )
        dataset["train"].to_csv(f"fpb_{FPB_CONFIGURATION}.csv")
        df = dataset["train"].to_pandas()
    df = df.rename(columns={"sentence": "phrase"})
    return df


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
    model: Annotated[str, typer.Argument(help="Model name. E.g. llama3.1 or mistral-nemo")],
    dataset: Annotated[str, typer.Argument(help="Dataset name. E.g. fpb")],
    limit: Annotated[
        Optional[int], typer.Option(help="Number of entries from dataset to analyze")
    ] = 10,
):
    """
    Use an LLM as an Agent to run sentiment analysis on a dataset with crewAI
    """
    if model not in OLLAMA_MODELS:
        print(f"Model {model} not supported")
        return
    model = f"ollama/{model}"
    if dataset not in DATASETS:
        print(f"Dataset {dataset} not supported")
        return

    print(f"Running sentiment analysis on dataset: {dataset} using {model}")
    if limit:
        print(f"Limiting to {limit} records")

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

    if dataset == "fpb":
        df = get_fpb_data()
        if limit:
            df = df.head(limit)
        records = df.to_dict(orient="records")
    else:
        print("Unsupported dataset")
        raise typer.Exit(1)

    print(f"Running sentiment analysis on {len(records)} records")
    print("Preview of dataset")
    df_preview = df.head(limit) if limit else df.head(10)
    print(tabulate(df_preview, headers="keys"))
    typer.confirm("Proceed with sentiment analysis?", abort=True)

    # Perform sentiment analysis
    results = crew.kickoff_for_each(inputs=records)

    outputs = [encode_sentiment(r.raw) for r in results]

    # Output result
    result_df = pd.DataFrame({"label": df["label"], "predicted": outputs})
    result_df["correct"] = result_df["predicted"] == result_df["label"]
    print("Accuracy: ", result_df["correct"].mean())
    print("Preview of dataset with predictions")
    print(tabulate(result_df, headers="keys"))
    result_df.to_csv("sa_results.csv", index=False)
    print("Results saved to sa_results.csv")


if __name__ == "__main__":
    typer.run(main)
