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
| 50 | 0.82s | 1.49s | 1.8x |
| 200 | 3.00s | 12.84s | 4.3x |
| 500 | 8.82s | 73.62s | 8.3x |

## Findings

- **The prediction held.** The speedup grew from 1.8x to 8.3x as generation went from 50 to 500 tokens.
- **With the cache, time grew about linearly.** From 200 to 500 tokens (2.5x), time grew 2.9x.
- **Without the cache, time grew about quadratically.** From 200 to 500 tokens, time grew 5.7x, close to the 6.25x that pure quadratic growth predicts.
- **Fixed costs matter at short lengths.** At 50 tokens, per-step overheads like running the layers hide most of the difference; as sequences grow, recomputation dominates.

This is why every production LLM uses a KV cache, and why systems like vLLM focus on managing its memory efficiently.

## Run it

```bash
pip install -r requirements.txt
python3 kv_cache_test.py
```
