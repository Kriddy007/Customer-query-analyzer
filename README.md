# Customer Query Analyzer

A simple and lightweight NLP pipeline that analyzes customer service queries using HuggingFace Transformers.

For a given query it returns:
- **Sentiment** — is the query positive or negative?
- **Intent** — what does the customer want or need?
- **Entities** — important things mentioned (organizations, people, locations)

## Models Used

| Task | Model |
|---|---|
| Sentiment Analysis | distilbert-base-uncased-finetuned-sst-2-english |
| Intent Classification | facebook/bart-large-mnli (zero-shot) |
| Named Entity Recognition | dbmdz/bert-large-cased-finetuned-conll03-english |

## Setup
```bash
pip install transformers torch
python Customer-query-analyser.py
```

## Sample Output
```
Query: My transaction failed but the money was deducted from my HDFC account
Sentiment => NEGATIVE (1.00)
Intent    => billing issue (0.87)
Entity    => ORG: HDFC (0.94)
```
