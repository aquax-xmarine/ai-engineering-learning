# Fitting a Line with PyTorch Autograd

Trains a straight line, y = wx + b, using only PyTorch tensors and autograd, with no `nn` modules or optimisers, to show how gradient descent and autograd work step by step.

## The idea

The data comes from the line y = 3x + 2, with random noise added. The model starts with w = 0 and b = 0 and isn't told the true values. It has to discover them from the data alone.

## How it works

Each training step follows the same loop used to train any neural network:

1. **Forward pass:** predict y using the current w and b. Autograd records every operation in its graph (DAG).
2. **Loss:** compute the mean squared error. The loss becomes the root of autograd's graph.
3. **Backward pass:** `loss.backward()` traces the graph from the loss back to w and b using the chain rule, and stores each gradient in `.grad`.
4. **Update:** move w and b a small step against their gradients, inside `torch.no_grad()` so the update isn't recorded.
5. **Reset:** clear the gradients with `.zero_()`, since autograd accumulates them.

## Results

| Step | Loss | w | b |
| --- | --- | --- | --- |
| 0 | 372.867 | 2.235 | 0.341 |
| 100 | 1.283 | 3.172 | 0.996 |
| 500 | 1.133 | 3.073 | 1.661 |
| 900 | 1.130 | 3.059 | 1.749 |

**Learned:** w = 3.059, b = 1.754 (true: 3, 2)

## Observations

- **w learns much faster than b.** In the very first step, w jumped from 0 to 2.235 while b only reached 0.341. The gradient for w is multiplied by x, which goes up to 10, so it's much larger.
- **w overshot, then corrected.** It passed 3 by step 100, then slowly came back down as b caught up.
- **The loss levelled off at about 1.13**, close to the noise's variance of 1. The model fitted everything it could; the rest is random noise it can't explain.
- **It didn't land exactly on 3 and 2.** The best-fit line for these 50 noisy points isn't exactly the true line, and b was still converging slowly. w and b also trade off: a slightly steeper slope with a lower intercept fits almost as well.

**Takeaway:** a model learns the best fit for the data it's given, noise included, not the true rule behind it.

## Run it

```bash
pip install -r requirements.txt
python3 line_fit.py
```