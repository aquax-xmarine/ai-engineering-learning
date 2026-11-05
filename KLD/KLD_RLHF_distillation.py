import numpy as np


def kl_divergence(p, q):
    """Compute KL(P || Q) for discrete distributions.

    p, q are arrays of probabilities that sum to 1.
    """
    # Avoid log(0) by filtering out zero probabilities
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


def kl_gaussians(mu, sigma_sq):
    """KL(N(mu, sigma_sq) || N(0, 1)) in closed form.

    The VAE KL penalty.
    """
    return 0.5 * (mu**2 + sigma_sq - np.log(sigma_sq) - 1)


# === Example 1: Two coins ===
P = np.array([0.7, 0.3])
Q = np.array([0.5, 0.5])

kl_pq = kl_divergence(P, Q)
kl_qp = kl_divergence(Q, P)

print("=== Two Coins ===")
print(f"KL(P || Q) = {kl_pq:.4f} nats = {kl_pq / np.log(2):.4f} bits")
print(f"KL(Q || P) = {kl_qp:.4f} nats = {kl_qp / np.log(2):.4f} bits")
print(f"Symmetric? {np.isclose(kl_pq, kl_qp)}")
print()

# Verify: KL = cross-entropy - entropy
H_P = -np.sum(P * np.log2(P))
H_PQ = -np.sum(P * np.log2(Q))

print(f"Entropy H(P) = {H_P:.4f} bits")
print(f"Cross-entropy H(P,Q) = {H_PQ:.4f} bits")
print(f"Difference = {H_PQ - H_P:.4f} bits")
print(f"KL(P||Q) in bits = {kl_pq / np.log(2):.4f} bits")
print(f"Match: {np.isclose(H_PQ - H_P, kl_pq / np.log(2))}")
print()

# === Example 2: VAE KL penalty ===
mu = 0.8
sigma_sq = 0.5

kl_vae = kl_gaussians(mu, sigma_sq)

print("=== VAE KL Penalty ===")
print(f"Encoder: mu={mu}, sigma²={sigma_sq}")
print(f"KL(encoder || prior) = {kl_vae:.4f} nats")
print()

# === Example 3: Softmax distributions (classification) ===
# Teacher vs student in knowledge distillation
teacher = np.array([0.05, 0.85, 0.10])  # confident: class 1
student = np.array([0.20, 0.60, 0.20])  # less confident

kl_distil = kl_divergence(teacher, student)

print("=== Knowledge Distillation ===")
print(f"Teacher: {teacher}")
print(f"Student: {student}")
print(f"KL(teacher || student) = {kl_distil:.4f} nats")
print()

# Trace it step by step
print("Step-by-step trace:")
for i, label in enumerate(["class_0", "class_1", "class_2"]):
    term = teacher[i] * np.log(teacher[i] / student[i])
    print(f"  {label}: {teacher[i]:.2f} × log({teacher[i]:.2f}/{student[i]:.2f}) "
          f"= {teacher[i]:.2f} × {np.log(teacher[i] / student[i]):.4f} = {term:.4f}")
print(f"  Sum = {kl_distil:.4f} nats")