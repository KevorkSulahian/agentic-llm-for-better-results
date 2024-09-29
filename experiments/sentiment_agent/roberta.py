import time
import warnings
from typing import Optional

import pandas as pd
import typer
import util
from config import BERT_MODELS, FPB_LABEL_MAP, TQDM_BAR_FORMAT, BertModulEnum, DataSetEnum
from rich import print
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing_extensions import Annotated

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_sentiment(tokenizer: AutoTokenizer, model, text: str) -> list[float]:
    """
    Use the model to predict the sentiment of the given text

    Args:
        tokenizer: Tokenizer that converts the text to tokens
        model: Model that predicts probabilities for each class for the tokenized text
        text: A financial text for sentiment analysis

    Returns:
        A list of probabilities for each class
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.logits.softmax(dim=1).tolist()[0]


def main(
    model_name: Annotated[
        BertModulEnum,
        typer.Argument(
            help="Model name",
            case_sensitive=False,
        ),
    ] = BertModulEnum.fin_distilroberta,
    dataset: Annotated[
        DataSetEnum, typer.Argument(help="Dataset name", case_sensitive=False)
    ] = DataSetEnum.fpb,
    limit: Annotated[
        Optional[int], typer.Option(help="Number of entries from dataset to analyze")
    ] = None,
    output: Annotated[Optional[str], typer.Option(help="Output file path")] = None,
    show: Annotated[bool, typer.Option(help="Show confusion matrix")] = False,
):
    """Use a BERT model to run sentiment analysis on a dataset"""
    df = util.get_fpb_data()
    if limit:
        df = df.head(limit)
    records = df.to_dict(orient="records")

    print(f"Running sentiment analysis on {len(records)} records")
    print("Preview of dataset")
    with pd.option_context("display.max_colwidth", 200):
        print(df.head(10))

    dist_df = util.get_class_distribution(df["label"].map(FPB_LABEL_MAP))
    print("Class distribution")
    with pd.option_context("display.float_format", "{:,.2f}".format):
        print(dist_df)
    print()

    typer.confirm("Proceed with sentiment analysis?", abort=True)

    # Load model and tokenizer
    hf_model_path = BERT_MODELS.get(model_name.value)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, cleaup_tokenization_spaces=False)
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_path)

    # Run the model
    start = time.time()
    results = []
    for item in tqdm(records, ncols=100, bar_format=TQDM_BAR_FORMAT):
        sentiment = get_sentiment(tokenizer, model, item["phrase"])
        results.append(sentiment)

    time_taken = time.time() - start
    util.print_time_taken(time_taken)

    results_df = pd.DataFrame(results, columns=[0, 1, 2]).round(2)

    # Set the predicted value equal to the label with the highest probability
    results_df.insert(0, "predicted", value=results_df.idxmax(axis=1))
    results_df["sentiment"] = results_df["predicted"].map(FPB_LABEL_MAP)
    results_df = results_df.rename(columns=FPB_LABEL_MAP)
    combined_df = pd.concat([df, results_df], axis=1)

    combined_df_preview = combined_df.head(limit) if limit else combined_df.head(10)
    pd.set_option("display.max_colwidth", 100)
    print(combined_df_preview)

    if output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output = f"roberta_{model_name.value}_{dataset.value}_{timestamp}.csv"

    combined_df.to_csv(output, index=False)
    print(f"Results saved to {output}")

    util.output_performance_summary(
        combined_df["label"], combined_df["predicted"], model_name.value, dataset.value, show
    )


if __name__ == "__main__":
    typer.run(main)
