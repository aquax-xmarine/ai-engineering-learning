# Cross-Encoder Basics

A minimal example of a cross-encoder scoring how relevant each document is to a query.

## What is a cross-encoder?

A cross-encoder reads the query and a document **together** as one input, so every query word can attend to every document word. It outputs a single relevance score for the pair. This makes it more accurate than a bi-encoder (like SBERT), which embeds query and document separately, but slower, so it's used to rerank a shortlist rather than search a whole collection.

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`, trained on MS MARCO web search data.

## Example

Query: *How do cross-encoders rerank documents?*

Four documents: two relevant, one loosely related (BM25), and one unrelated (Kathmandu).

## Results

```
(paste your output here)
```

## Observations

- The relevant documents should score highest, and the unrelated one lowest.
- The scores are raw relevance scores (logits), not limited to 0–1, so only their order matters.

## Run it

```bash
pip install -r requirements.txt
python3 rerank_test.py
```