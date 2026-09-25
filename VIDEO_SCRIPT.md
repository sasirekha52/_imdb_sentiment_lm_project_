# 3–5 Minute Demo Video Script

## 0:00–0:30 — Introduction

"Hello, this is my Advanced Language Model project. I built an IMDb Sentiment Intelligence system using DistilBERT. The goal is to classify movie reviews as positive or negative."

## 0:30–1:00 — Dataset

"The project uses the provided IMDb Dataset.csv. It contains review text and sentiment labels. I first validate the schema, remove invalid or duplicate records, inspect class balance, and create reproducible stratified train, validation, and test sets."

## 1:00–1:30 — Baseline

"Before using a Transformer, I created a TF-IDF plus Logistic Regression baseline. This gives me a reference point for evaluating whether contextual language modeling adds useful performance."

## 1:30–2:15 — Language model

"For the language model, I selected DistilBERT. It provides contextual Transformer representations while being smaller than full BERT. I tokenize the reviews, truncate them to a controlled maximum length, dynamically pad batches, and fine-tune the sequence-classification head."

## 2:15–3:00 — Evaluation

"After training, I evaluate the model on the held-out test set using accuracy, precision, recall, F1-score, and a confusion matrix. I also compare the Transformer with the classical baseline."

## 3:00–3:45 — Error analysis and inference

"I then inspect incorrect predictions, especially high-confidence errors. Finally, I test new movie reviews using the saved model and show the predicted sentiment and confidence."

## 3:45–4:30 — Real-world considerations

"The project also includes inference latency measurement and a Streamlit interface. In a production environment, I would add monitoring, model versioning, drift detection, privacy controls, and periodic evaluation."

## Important

Read the actual metric values displayed by your notebook during the recording. Do not claim a metric that you did not obtain from your own run.
