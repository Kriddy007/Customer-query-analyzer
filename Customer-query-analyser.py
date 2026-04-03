from transformers import pipeline

ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

sentiment = pipeline("sentiment-analysis",model="distilbert-base-uncased-finetuned-sst-2-english")

intent = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

queries = [
    "My transaction failed but the money was deducted from my HDFC account",
    "I have been waiting 3 days for my refund of 5000 rupees",
    "John Smith from Mumbai wants to upgrade his account",
    "My account got locked and nobody is helping me since Monday",
    "I got charged twice for my order placed on Amazon Pay"
]

intents = ["billing issue", "account management", "technical support", "refund request", "general inquiry"]


def analyze_query(query):
  print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
  entities = ner(query)
  print(f"Query: {query}")

  sentiment_query=sentiment(query)[0]
  print(f"Sentiment => {sentiment_query['label']} {sentiment_query['score']:.2f}\n")

  intent_query=intent(query , candidate_labels=intents)
  print(f"→ Intent: {intent_query['labels'][0]} ({intent_query['scores'][0]:.2f})\n")

  for entity in entities:
      print(f"{entity['entity_group']}: {entity['word']} ({entity['score']:.2f})\n")
  print()
  print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


for q in queries:
    analyze_query(q)