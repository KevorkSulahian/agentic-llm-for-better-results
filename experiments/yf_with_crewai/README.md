# Financial Agents with crewAI

The following experiment is inspired by the [Medium post by author Batuhan Sener](https://medium.com/@batuhansenerr/ai-powered-financial-analysis-multi-agent-systems-transform-data-into-insights-d94e4867d75d).
The crew structure is similar, but the experiment tries to use Yahoo Finance News instead of Reddit posts.

The experiment incorporates a sentiment analysis (SA) tool that uses a pretrained transformers model that is based on RoBERTa.
The SA tool fetches the latest news from RSS Yahoo Finance News, and labels each summary of the news articles with a sentiment
label "positive", "negative" or "neutral".

HuggingFace model for sentiment analysis: [mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)

The code is also adjusted to include the possibility to use a local hosted LLM with [ollama](https://ollama.com/).
