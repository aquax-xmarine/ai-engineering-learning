# KL Divergence: From Coins to RLHF and Distillation

Computes Kullback–Leibler (KL) divergence with NumPy, and shows where it appears in machine learning: entropy and cross-entropy, VAEs, and knowledge distillation.

## What is KL divergence?

KL divergence measures how different one probability distribution is from another:

```
D_KL(P || Q) = Σₓ P(x) log( P(x) / Q(x) )
```

Intuitively: how many extra bits does Q waste because it doesn't know the true distribution P?

## Functions

| Function | What it computes |
| --- | --- |
| `kl_divergence(p, q)` | KL(P ‖ Q) for discrete distributions, skipping outcomes where P is 0 to avoid log(0) |
| `kl_gaussians(mu, sigma_sq)` | KL between a Gaussian N(μ, σ²) and the standard normal N(0, 1), in closed form, the penalty used in VAEs |

## Examples and results

### 1. Two coins

A true coin P (70% heads) and a model Q that assumes a fair coin (50/50).

```
KL(P || Q) = 0.0823 nats = 0.1187 bits
KL(Q || P) = 0.0872 nats = 0.1258 bits
Symmetric? False
```

The script also verifies that KL equals cross-entropy minus entropy:

```
Entropy H(P)         = 0.8813 bits
Cross-entropy H(P,Q) = 1.0000 bits
Difference           = 0.1187 bits
Match: True
```

### 2. VAE KL penalty

An encoder outputs μ = 0.8 and σ² = 0.5. Its KL divergence from the standard normal prior:

```
KL(encoder || prior) = 0.4166 nats
```

### 3. Knowledge distillation

A confident teacher and a less confident student, over three classes:

```
Teacher: [0.05 0.85 0.10]
Student: [0.20 0.60 0.20]
KL(teacher || student) = 0.1574 nats
```

Step-by-step trace:

```
class_0: 0.05 × log(0.05/0.20) = 0.05 × -1.3863 = -0.0693
class_1: 0.85 × log(0.85/0.60) = 0.85 ×  0.3483 =  0.2961
class_2: 0.10 × log(0.10/0.20) = 0.10 × -0.6931 = -0.0693
Sum = 0.1574 nats
```

## Observations

- **KL is not symmetric.** KL(P‖Q) = 0.0823 but KL(Q‖P) = 0.0872, so the direction matters. This is why distillation and RLHF have to choose which way to measure it.
- **KL = cross-entropy − entropy.** That's why minimising cross-entropy loss in classification is equivalent to minimising KL against the true labels.
- **Individual terms can be negative.** In the distillation trace, classes 0 and 2 contribute negative values, but the total is always zero or more, and only zero when the two distributions are identical.
- **Most of the distillation KL comes from the teacher's top class.** The student puts 0.60 on class 1 where the teacher puts 0.85, and that gap contributes almost all of the 0.1574.

## Where KL divergence is used

- **Classification:** cross-entropy loss
- **Knowledge distillation:** training a student to match a teacher's output distribution
- **RLHF:** a KL penalty keeps the fine-tuned policy close to the reference model
- **VAEs:** a KL penalty keeps the learned latent distribution close to the prior

## Run it

```bash
pip install -r requirements.txt
python3 kl_divergence.py
```