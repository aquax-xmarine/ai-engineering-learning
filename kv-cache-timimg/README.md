# KV Cache Timing Experiment

Measures how much the KV cache speeds up text generation in GPT-2.

## What is the KV cache?

When a model generates text, each new token attends to all previous tokens. Without a cache, the keys (K) and values (V) of every earlier token are recomputed at each step. The KV cache stores them, so each step only computes K and V for the one new token.

## Prediction

At short lengths, the difference would be small. As the number of generated tokens grows, the speedup should grow too: without the cache, total work grows roughly quadratically, and with it, roughly linearly.

## Results

GPT-2 (124M) on CPU, greedy decoding.

| Tokens generated | With cache | Without cache | Speedup |
| --- | --- | --- | --- |
| 50 | 0.73s | 1.40s | 1.9x |
| 200 | 3.20s | 12.16s | 3.8x |

## Findings

- **The prediction held.** The speedup doubled from 1.9x to 3.8x as generation grew from 50 to 200 tokens.
- **With the cache, time grew about linearly:** 4x more tokens took 4.4x longer.
- **Without the cache, time grew much faster:** 4x more tokens took 8.7x longer.
- **Not fully quadratic yet.** Pure quadratic growth would be 16x. GPT-2 is small, so at these lengths fixed per-step costs, like running the layers, take much of the time. The quadratic recomputation dominates more at longer lengths.

This is why every production LLM uses a KV cache, and why systems like vLLM focus on managing its memory efficiently.

## Run it

```bash
pip install -r requirements.txt
python3 kv_cache_test.py
```