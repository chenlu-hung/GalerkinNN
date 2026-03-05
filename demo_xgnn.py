"""
Demo script for xGNN regression with three test problems.
All problems include additive Gaussian noise.

Test 1: f(x) = sin(3x) + 0.5*cos(7x) + noise, no knowledge-based functions.
Test 2: f(x) = 2*sin(5x) + x^0.7 + 0.3*tanh(x-2) + noise, with knowledge-based templates.
Test 3: f(x) = |x-pi|^0.5 * sign(x-pi) + sawtooth-like sharp features + noise.
"""

import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Force use of CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NOISE_LEVEL = 0.
# Generator must also be on the same device as tensors
GEN = torch.Generator(device=DEVICE).manual_seed(42)

# ---- Test 1: Smooth function + noise, no knowledge-based functions -------

def test1():
    print("\n" + "=" * 60)
    print("Test 1: f(x) = sin(3x) + 0.5*cos(7x) + noise")
    print("=" * 60)

    x = torch.linspace(0, 2 * torch.pi, 200, device=DEVICE)
    y_true = torch.sin(3 * x) + 0.5 * torch.cos(7 * x)
    y = y_true + NOISE_LEVEL * torch.randn(x.shape[0], generator=GEN, device=DEVICE)

    model = xgnn_regression(
        X=x,
        y=y,
        tol=1e-5,
        max_iter=200,
        net_factory=lambda d: SmallNetwork(input_dim=d, width=128, depth=3).to(DEVICE),
        lr=1e-2,
        num_epochs=500,
        num_restarts=3,
        device=DEVICE
    )

    y_pred = model.predict(x).detach().cpu()
    print(f"  Number of basis functions: {len(model.basis_functions)}")
    print(f"  Max error vs true: {torch.max(torch.abs(y_true.cpu() - y_pred)):.6f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x.cpu().numpy(), y.cpu().numpy(), ".", color="gray", markersize=3, label="Noisy data")
    ax.plot(x.cpu().numpy(), y_true.cpu().numpy(), "k-", lw=2, label="Ground truth")
    ax.plot(x.cpu().numpy(), y_pred.numpy(), "r--", lw=1.5, label="xGNN prediction")
    ax.set_title("Test 1: sin(3x) + 0.5·cos(7x) + noise")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    fig.tight_layout()
    plt.show()


# ---- Test 2: Function with known structure + knowledge-based templates ---

def test2():
    print("\n" + "=" * 60)
    print("Test 2: f(x) = 2*sin(5x) + x^0.7 + 0.3*tanh(x-2) + noise")
    print("        Knowledge-based: sin(mu*x), x^mu")
    print("=" * 60)

    x = torch.linspace(0.1, 5, 200, device=DEVICE)
    y_true = 2 * torch.sin(5 * x) + x ** 0.7 + 0.3 * torch.tanh(x - 2)
    y = y_true + NOISE_LEVEL * torch.randn(x.shape[0], generator=GEN, device=DEVICE)

    def sin_template(X: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        return torch.sin(mu * X.squeeze(-1))

    def power_template(X: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        return X.squeeze(-1).abs().pow(mu)

    phi_templates = [
        (sin_template, 1.0),
        (power_template, 1.0),
    ]

    model = xgnn_regression(
        X=x,
        y=y,
        tol=1e-5,
        max_iter=200,
        net_factory=lambda d: SmallNetwork(input_dim=d, width=128, depth=3).to(DEVICE),
        phi_templates=phi_templates,
        lr=1e-2,
        num_epochs=500,
        num_restarts=3,
        device=DEVICE
    )

    y_pred = model.predict(x).detach().cpu()
    print(f"  Number of basis functions: {len(model.basis_functions)}")
    print(f"  Max error vs true: {torch.max(torch.abs(y_true.cpu() - y_pred)):.6f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x.cpu().numpy(), y.cpu().numpy(), ".", color="gray", markersize=3, label="Noisy data")
    ax.plot(x.cpu().numpy(), y_true.cpu().numpy(), "k-", lw=2, label="Ground truth")
    ax.plot(x.cpu().numpy(), y_pred.numpy(), "r--", lw=1.5, label="xGNN prediction")
    ax.set_title("Test 2: 2·sin(5x) + x^0.7 + 0.3·tanh(x−2) + noise")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    fig.tight_layout()
    plt.show()


# ---- Test 3: Sharp derivatives (kink + step + cusp) + noise -------------

def test3():
    print("\n" + "=" * 60)
    print("Test 3: Sharp-derivative function + noise")
    print("        f(x) = |x-2| + 2*step(x-4) + sqrt(|x-6|)")
    print("=" * 60)

    x = torch.linspace(0, 8, 400, device=DEVICE)
    y_true = torch.abs(x - 2) + 2.0 * (x >= 4).float() + torch.sqrt(torch.abs(x - 6))
    y = y_true + NOISE_LEVEL * torch.randn(x.shape[0], generator=GEN, device=DEVICE)

    model = xgnn_regression(
        X=x,
        y=y,
        tol=1e-5,
        max_iter=200,
        net_factory=lambda d: SmallNetwork(input_dim=d, width=128, depth=3).to(DEVICE),
        lr=1e-2,
        num_epochs=500,
        num_restarts=3,
        device=DEVICE
    )

    y_pred = model.predict(x).detach().cpu()
    print(f"  Number of basis functions: {len(model.basis_functions)}")
    print(f"  Max error vs true: {torch.max(torch.abs(y_true.cpu() - y_pred)):.6f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x.cpu().numpy(), y.cpu().numpy(), ".", color="gray", markersize=3, label="Noisy data")
    ax.plot(x.cpu().numpy(), y_true.cpu().numpy(), "k-", lw=2, label="Ground truth")
    ax.plot(x.cpu().numpy(), y_pred.numpy(), "r--", lw=1.5, label="xGNN prediction")
    ax.set_title("Test 3: |x−2| + 2·H(x−4) + √|x−6| + noise")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    fig.tight_layout()
    plt.show()

def custom_net(input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Flatten(0),
        ).to(DEVICE)

if __name__ == "__main__":
    test1()
    test2()
    test3()