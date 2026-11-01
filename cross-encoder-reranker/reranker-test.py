from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "How do cross-encoders rerank documents?"
docs = [
    "A cross-encoder reads the query and document together and outputs a relevance score.",
    "BM25 is a keyword-based ranking function used in search engines.",
    "Kathmandu is the capital city of Nepal.",
    "Rerankers reorder retrieved results to put the most relevant ones first.",
]

scores = model.predict([(query, doc) for doc in docs])

for score, doc in sorted(zip(scores, docs), reverse=True):
    print(f"{score:.3f}  {doc}")

