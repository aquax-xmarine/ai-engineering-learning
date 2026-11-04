# Self-Attention from Scratch in NumPy

A simplified self-attention mechanism built with only NumPy, following the blog post [Building a Simplified Self-Attention Mechanism in Python](https://patotricks15.medium.com/building-a-simplified-self-attention-mechanism-in-python-748ee8909b41), to understand the core idea behind transformers.

## What is self-attention?

Self-attention lets each word in a sentence decide how much attention to give every other word. Each word is represented by three vectors:

- **Query (Q):** what the word is looking for
- **Key (K):** what each word offers
- **Value (V):** the information a word passes on if it's chosen

## How it works

1. **Embeddings:** each word gets a 4-dimensional vector.
2. **Attention scores:** compare every query with every key, `Q · Kᵀ`.
3. **Softmax:** turn the scores into attention weights.
4. **Weighted sum:** combine the values using those weights.

```
weighted values = softmax(Q · Kᵀ) · V
```

The script then prints the top 3 words each word attends to, and the word with the highest attention other than itself.

## Known limitations

This is a deliberately simplified version. Compared with attention in real transformers:

- **Softmax is applied column-wise** (`axis=0`). In real attention it's applied row-wise, so each word's weights over all other words add up to 1.
- **Q, K and V are identical.** Real models multiply the embeddings by three learned matrices, `W_q`, `W_k` and `W_v`, so each word gets three different views.
- **No scaling by √dₖ.** The full formula is `softmax(Q · Kᵀ / √dₖ) · V`. Without scaling, large dot products make softmax too peaked.
- **Embeddings are random.** They don't reflect word meanings and change every run, so the attention patterns are illustrative only.
- **The explanation assumes position 0 is the word itself** after sorting, which isn't guaranteed. And each row shows which words a word *attends to*, so "is most attended by" reads in the reverse direction.

## Run it

```bash
pip install -r requirements.txt
python3 self_attention.py
```