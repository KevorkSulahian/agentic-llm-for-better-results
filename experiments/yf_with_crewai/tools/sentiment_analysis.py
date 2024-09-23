# sentiment_analysis_tool.py

import feedparser

# import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from crewai_tools import tool

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
)


def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text.
    Returns:
        dict: Dictionary of labels and their scores.
    """
    # inputs = tokenizer(text, return_tensors="pt")
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,  # Truncate sequences longer than max_length
        max_length=512,  # Set the maximum length to 512 tokens
    )
    outputs = model(**inputs)
    scores = outputs.logits.softmax(dim=1)[0]
    labels = ["negative", "neutral", "positive"]
    scores_dict = dict(zip(labels, scores.tolist()))
    return scores_dict


@tool
def ticker_sentiment_analysis(ticker: str, keyword: str):
    """
    Perform sentiment analysis on Yahoo Finance RSS feed for a specific ticker and keyword.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'META').
        keyword (str): Keyword to filter articles (e.g., 'meta').

    Returns:
        dict: Final score, overall sentiment, number of articles, and sentiment counts.
    """
    rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    feed = feedparser.parse(rss_url)

    total_score = 0
    num_articles = 0
    sentiments_counts = {"neutral": 0, "negative": 0, "positive": 0}

    for entry in feed.entries:
        # Check if the keyword is in the article title or summary
        if (
            keyword.lower() not in entry.summary.lower()
            and keyword.lower() not in entry.title.lower()
        ):
            continue

        text = entry.summary
        # Perform sentiment analysis
        scores = analyze_sentiment(text)
        sentiment_label = max(scores, key=scores.get)
        sentiment_score = scores["positive"] - scores["negative"]

        # Update counts and total score
        sentiments_counts[sentiment_label] += 1
        total_score += sentiment_score
        num_articles += 1

    final_score = total_score / num_articles if num_articles > 0 else 0
    overall_sentiment = (
        "positive" if final_score > 0.15 else "negative" if final_score < -0.15 else "neutral"
    )

    return {
        "ticker": ticker,
        "final_score": final_score,
        "overall_sentiment": overall_sentiment,
        "num_articles": num_articles,
        "sentiments_counts": sentiments_counts,
    }


# ticker = 'META'
# keyword = 'meta'

# result = ticker_sentiment_analysis(ticker, keyword)
# print(f"Final score for {ticker}: {result['final_score']}")
# print(f"Overall sentiment: {result['overall_sentiment']} ({result['final_score']})")
# print(f"Number of articles analyzed: {result['num_articles']}")
# print(f"Sentiment counts: {result['sentiments_counts']}")
