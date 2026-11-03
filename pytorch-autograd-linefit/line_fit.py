import torch

torch.manual_seed(0)

# Data: y = 3x + 2, plus noise
x = torch.linspace(0, 10, 50)
y = 3 * x + 2 + torch.randn(50)

# Parameters to learn, starting at zero
w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.01

for step in range(1000):
    pred = w * x + b                    # forward pass: builds the graph
    loss = ((pred - y) ** 2).mean()     # mean squared error, the root of the DAG

    loss.backward()                     # backward pass: fills w.grad and b.grad

    with torch.no_grad():               # don't record the update in the graph
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()                      # clear gradients so they don't accumulate
    b.grad.zero_()

    if step % 100 == 0:
        print(f"step {step:4d} | loss {loss.item():.3f} | w {w.item():.3f} | b {b.item():.3f}")

print(f"\nLearned: w = {w.item():.3f}, b = {b.item():.3f}  (true: 3, 2)")