from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import FPB_CONFIGURATION
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SECONDS_IN_HOUR = 3600
SECONDS_IN_MINUTE = 60


def get_fpb_data() -> pd.DataFrame:
    """
    Loads the Financial PhraseBank dataset from Hugging Face.
    The dataset is cached to avoid downloading it multiple times.
    https://huggingface.co/datasets/takala/financial_phrasebank

    Returns:
        DataFrame with the dataset
    """
    filename = f"fpb_{FPB_CONFIGURATION}.csv"
    if Path.is_file(Path(filename)):
        print(f"Using cached FPB dataset, stored at {filename}")
        df = pd.read_csv(filename)
    else:
        print("Downloading FPB dataset")
        dataset = load_dataset(
            path="takala/financial_phrasebank",
            name=FPB_CONFIGURATION,
            trust_remote_code=True,
        )
        dataset["train"].to_csv(filename)
        df = dataset["train"].to_pandas()
    df = df.rename(columns={"sentence": "phrase"})
    return df


def get_class_distribution(series: pd.Series) -> pd.DataFrame:
    """
    Returns the class distribution of the dataset.
    """
    value_counts = series.value_counts()
    value_counts_df = pd.DataFrame(
        {"count": value_counts, "normalized": value_counts / value_counts.sum()}
    )

    # Add the total row
    total_row = pd.DataFrame({"count": [value_counts.sum()], "normalized": [1]}, index=["Total"])
    value_counts_df = pd.concat([value_counts_df, total_row])
    return value_counts_df


def plot_confusion_matrix(actual: pd.Series, predicted: pd.Series, show: bool = False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(actual, predicted, normalize="all")
    num_classes = actual.nunique()
    labels = (
        np.asarray(
            ["{0:.1%}\n{1:d}".format(value, int(value * len(actual))) for value in cm.flatten()]
        )
    ).reshape(num_classes, num_classes)
    sns.heatmap(
        cm,
        # annot=True,
        annot=labels,
        fmt="",
        cmap="Blues",
        square=True,
        xticklabels=["negative", "neutral", "positive"],
        yticklabels=["negative", "neutral", "positive"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if show:
        plt.show()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"confusion_matrix_{timestamp}.png")
    print(f"Confusion matrix saved to confusion_matrix_{timestamp}.png")


def output_performance_summary(
    actual: pd.Series, predicted: pd.Series, model_name: str, dataset: str, show: bool = False
):
    """Output performance summary of the model"""
    report = classification_report(
        actual,
        predicted,
        target_names=["negative", "neutral", "positive"],
    )
    acc_score = accuracy_score(actual, predicted)

    print("Classification report")
    print(report)
    print(f"Classification accuracy is: {acc_score:.2f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"classification_report_{model_name}_{dataset}_{timestamp}.txt"
    with open(output_file, "w") as f:
        f.write("Classification report\n")
        f.write(report)
        f.write(f"Classification accuracy is: {acc_score:.2f}\n")

    plot_confusion_matrix(actual, predicted, show)


def print_time_taken(seconds: float):
    tdelta = timedelta(seconds=seconds)

    if tdelta.seconds >= SECONDS_IN_HOUR:
        hours, remainder = divmod(tdelta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Time taken: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    elif tdelta.seconds >= SECONDS_IN_MINUTE:
        minutes, seconds = divmod(tdelta.seconds, 60)
        print(f"Time taken: {int(minutes)}m {seconds:.2f}s")
    else:
        print(f"Time taken: {tdelta.seconds}s")
