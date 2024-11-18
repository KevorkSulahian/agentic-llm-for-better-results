# Large Language Models

In the FinMAS project, we have mainly used the llama3-8b and gpt-4o models.
A summary of their key features is provided below.

| Model Id       | Context Window | Parameter Size | Released   | Input Cost ($/MT) | Output Cost ($/MT) |
| -------------- | -------------- | -------------- | ---------- | ----------------- | ------------------ |
| gpt-4o-mini    | 128k           | Not disclosed  | 2024-07-18 | 0.15              | 0.6                |
| gpt-4o         | 128k           | Not disclosed  | 2024-05-13 | 2.5               | 10                 |
| llama3-8b-8192 | 8192           | 8b             | 2024-04-18 | Free              | Free               |

$/MT = Cost per million tokens

## OpenAI gpt-4o and gpt-4o-mini

The gpt-4o models from OpenAI are currently the main models that are available for developers at a
decent cost. The gpt-4o-mini model is particularly cost-effective and is a good choice general-purpose
tasks.

## Groq hosted models

It is possible to use other models hosted by Groq than the llama3-8b-8192 model, however
within the free tier this model have the most capacity in terms of tokens per minute that
the model is able to process. For trying other models, find the
[model id in their documentation](https://console.groq.com/docs/models).
