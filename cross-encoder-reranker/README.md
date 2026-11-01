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
 2.312  A cross-encoder reads the query and document together and outputs a relevance score.
-4.167  Rerankers reorder retrieved results to put the most relevant ones first.
-11.386  BM25 is a keyword-based ranking function used in search engines.
-11.431  Kathmandu is the capital city of Nepal.
```

## Observations

- **The ranking is sensible.** The document that directly answers how cross-encoders work scores highest, and the unrelated Kathmandu sentence scores lowest.
- **Scores are raw logits, not probabilities.** They can be negative and aren't limited to 0–1; only their order matters. Applying a sigmoid turns them into probabilities: 2.312 → about 0.91, and −4.167 → about 0.02.
- **It judges answers, not topics.** The rerankers sentence is on topic but doesn't explain *how* cross-encoders work, so it scores far lower than the top document. The model was trained on MS MARCO, where it learned to reward passages that actually answer the question.
- **BM25 scores as low as Kathmandu.** Even though BM25 relates to search, the model treats it as just as irrelevant as a fact about Nepal (−11.39 vs −11.43). A bi-encoder would likely see BM25 as much closer, because the topics overlap. This shows the difference between relevance and topical similarity.

## Run it

```bash
pip install -r requirements.txt
python3 rerank_test.py
```
