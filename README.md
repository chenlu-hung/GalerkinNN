# GalerkinNN

A PyTorch implementation of the Extended Galerkin Neural Network (xGNN) for least-squares regression, based on the framework of Ainsworth & Dong.

## Overview

The xGNN algorithm iteratively constructs a solution by:

1. Training small neural network basis functions to maximise correlation with current residuals.
2. Optionally augmenting each basis with knowledge-based parametric functions (e.g., `sin(mu*x)`, `x^mu`).
3. Refitting all coefficients jointly via least squares after each new basis is added.

This greedy approach builds an interpretable, compact representation of the target function with controllable accuracy.

## Files

- **`xgnn_regression.py`** — Core library implementing the xGNN regression algorithm (`xgnn_regression()`, `XGNNModel`, `SmallNetwork`, etc.).
- **`demo_xgnn.py`** — Demo script with three test problems showcasing smooth functions, knowledge-augmented regression, and sharp-feature approximation.

## Quick Start

```python
import torch
from xgnn_regression import xgnn_regression

x = torch.linspace(0, 6.28, 200)
y = torch.sin(3 * x) + 0.5 * torch.cos(7 * x)

model = xgnn_regression(X=x, y=y, tol=1e-5, max_iter=50, device="cuda")
y_pred = model.predict(x)
```

## Requirements

- Python 3.10+
- PyTorch (with CUDA recommended)
- matplotlib (for demos)

## References

1. M. Ainsworth and J. Dong, "Galerkin Neural Networks: A Framework for Approximating Variational Equations with Error Control," *SIAM Journal on Scientific Computing*, vol. 43, no. 4, pp. A2474–A2501, 2021.

2. M. Ainsworth and J. Dong, "Extended Galerkin Neural Network Approximation of Singular Variational Problems with Error Control," *SIAM Journal on Scientific Computing*, vol. 47, no. 3, pp. C738–C768, 2025. DOI: [10.1137/24M1658279](https://doi.org/10.1137/24M1658279)
