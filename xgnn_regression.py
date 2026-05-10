"""
Extended Galerkin Neural Network (xGNN) for Least-Squares Regression.

Faithful regression specialisation of the xGNN framework from
Ainsworth & Dong (2024). For a(u, v) = (u, v) and F(v) = (y, v):

  * Each iteration trains (W, b, μ) by gradient ascent on ||proj_B r||,
    where B = [h(·) | Ψ(·; μ)] are the activation atoms; the linear
    coefficients [c; d] are obtained by an inner least-squares solve at
    every gradient step (eq. 2.44-2.45).
  * The outer Galerkin solve uses separate coefficients for the σ-part
    and Ψ-part of each basis function (eq. 2.30-2.31).
  * An optional initial approximation u_0 may be supplied.
  * Stopping uses the a posteriori estimator <r, φ_i> (Cor. 2.6,
    eq. 2.37), which equals ||proj_B r|| at the inner LSQ optimum
    under the normalised basis function convention of the paper.
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hidden-only feedforward network
# ---------------------------------------------------------------------------

class SmallNetwork(nn.Module):
    """Feedforward network producing hidden-unit activations h(x) of shape (N, width).

    The output linear coefficients c (eq. 2.6 / 2.45) are NOT part of this
    module — they are obtained by the inner least-squares solve at each
    gradient step.
    """

    def __init__(self, input_dim: int, width: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.Tanh())
            in_features = width
        self.hidden = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden(x)


# ---------------------------------------------------------------------------
# Basis function wrapper
# ---------------------------------------------------------------------------

@dataclass
class BasisFunction:
    """Stores trained NN, frozen Ψ params, and inner-LSQ coefficients (c, d).

    φ_σ(x) = c · h(x);   φ_Ψ(x) = Σ_ℓ d_ℓ Ψ_ℓ(x; μ_ℓ).
    """

    network: nn.Module
    c: torch.Tensor                                                   # (width,)
    psi_funcs: list[tuple[Callable, nn.Parameter]] = field(default_factory=list)
    d: Optional[torch.Tensor] = None                                  # (m_*,) or None

    def evaluate_sigma(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.network(X) @ self.c

    def evaluate_psi(self, X: torch.Tensor) -> torch.Tensor:
        if not self.psi_funcs or self.d is None:
            return torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
        with torch.no_grad():
            atoms = torch.column_stack([fn(X, mu) for fn, mu in self.psi_funcs])
            return atoms @ self.d

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        return self.evaluate_sigma(X) + self.evaluate_psi(X)


# ---------------------------------------------------------------------------
# Inner LSQ objective
# ---------------------------------------------------------------------------

def _build_atom_matrix(
    network: nn.Module,
    psi_funcs: list[tuple[Callable, nn.Parameter]],
    X: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Stack hidden activations and Ψ atoms into B; return (B, n_hidden)."""
    h = network(X)
    n_hidden = h.shape[1]
    if psi_funcs:
        atoms = torch.column_stack([fn(X, mu) for fn, mu in psi_funcs])
        B = torch.cat([h, atoms], dim=1)
    else:
        B = h
    return B, n_hidden


def _ridge_solve(B: torch.Tensor, r: torch.Tensor, ridge: float) -> torch.Tensor:
    """Solve (BᵀB + α·I) x = Bᵀr with α relative to ||B||² and adaptive escalation.

    Using regularised normal equations rather than `torch.linalg.lstsq`
    keeps backward off the SVD path (which fails to converge on CUDA when
    B is near rank-deficient, e.g. saturated tanh units). `ridge` is
    interpreted as a *relative* tolerance — it is multiplied by the mean
    diagonal of BᵀB so the same value works regardless of input/sample
    scale. The ridge is also floored at a dtype-aware value (a small
    constant multiple of machine epsilon, *independent of p_cols*) so the
    floor does not strengthen as the basis grows. If the solve still
    reports singular (very rank-deficient B), α is escalated by 10× before
    giving up.
    The whole solve is performed in float64 regardless of the caller's
    dtype: forming BᵀB squares cond(B), and float32 normal equations
    bottom out at √eps·cond(B) ≈ 3e-4·cond(B), which is the dominant
    precision floor here. Upcasting moves that floor to ~eps·cond(B) ≈
    2e-16·cond(B). The cast is differentiable, so autograd still flows
    back into the (typically float32) network parameters.
    """
    out_dtype = B.dtype
    B = B.to(torch.float64)
    r = r.to(torch.float64)
    BtB = B.T @ B
    Btr = B.T @ r
    eye = torch.eye(B.shape[1], device=B.device, dtype=B.dtype)
    scale = BtB.diagonal().abs().mean().detach().clamp_min(1.0)
    eps = torch.finfo(B.dtype).eps
    alpha = torch.maximum(
        torch.as_tensor(ridge, device=B.device, dtype=B.dtype) * scale,
        torch.as_tensor(eps, device=B.device, dtype=B.dtype) * scale,
    )
    for _ in range(12):
        solution, info = torch.linalg.solve_ex(
            BtB + alpha * eye, Btr, check_errors=False
        )
        if int(info.max().item()) == 0 and bool(torch.isfinite(solution).all().item()):
            return solution.to(out_dtype)
        alpha = alpha * 10.0
    raise torch.linalg.LinAlgError(
        f"adaptive ridge solve failed; final alpha={alpha.detach().cpu().item():.2e}"
    )


def _inner_objective(
    network: nn.Module,
    psi_funcs: list[tuple[Callable, nn.Parameter]],
    X: torch.Tensor,
    residuals: torch.Tensor,
    ridge: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Inner LSQ for [c; d]; returns (proj_norm, coeffs, n_hidden).

    With [c; d] ≈ argmin ||B[c;d] - r||², the projection onto col(B) is
    P_B r = B[c;d], and the maximiser of ⟨r, v⟩/|||v||| over span(B) has
    value ||P_B r||.
    """
    B, n_hidden = _build_atom_matrix(network, psi_funcs, X)
    coeffs = _ridge_solve(B, residuals, ridge)
    proj_norm = (B @ coeffs).norm()
    return proj_norm, coeffs, n_hidden


# ---------------------------------------------------------------------------
# Training a single basis function
# ---------------------------------------------------------------------------

def train_basis_function(
    X: torch.Tensor,
    residuals: torch.Tensor,
    net_factory: Callable[[], nn.Module],
    phi_templates: list[tuple[Callable, float]],
    lr: float,
    num_epochs: int,
    num_restarts: int = 3,
    device: torch.device | str = "cuda",
    ridge: float = 1e-8,
) -> tuple[
    nn.Module,
    torch.Tensor,
    list[tuple[Callable, nn.Parameter]],
    Optional[torch.Tensor],
    float,
]:
    """Train (W, b, μ) by gradient ascent with inner LSQ for [c; d].

    Returns (best_network, best_c, best_psi_funcs, best_d, best_proj_norm).
    """
    best_proj_norm = -float("inf")
    best_network = None
    best_c = None
    best_psi_funcs = None
    best_d = None

    for _ in range(num_restarts):
        net = net_factory().to(device)

        psi_funcs: list[tuple[Callable, nn.Parameter]] = []
        for fn, mu_init in phi_templates:
            mu = nn.Parameter(
                torch.tensor(mu_init, dtype=torch.float32, device=device)
            )
            psi_funcs.append((fn, mu))

        params: list[torch.Tensor] = list(net.parameters())
        for _, mu in psi_funcs:
            params.append(mu)

        optimizer = torch.optim.Adam(params, lr=lr)

        for _epoch in range(num_epochs):
            optimizer.zero_grad()
            proj_norm, _, _ = _inner_objective(
                net, psi_funcs, X, residuals, ridge=ridge
            )
            (-proj_norm).backward()
            optimizer.step()

        with torch.no_grad():
            proj_norm_t, coeffs, n_hidden = _inner_objective(
                net, psi_funcs, X, residuals, ridge=ridge
            )
        final_proj = proj_norm_t.item()

        if final_proj > best_proj_norm:
            best_proj_norm = final_proj
            best_network = copy.deepcopy(net)
            best_c = coeffs[:n_hidden].detach().clone()
            best_psi_funcs = [
                (fn, nn.Parameter(mu.detach().clone()))
                for fn, mu in psi_funcs
            ]
            best_d = (
                coeffs[n_hidden:].detach().clone() if psi_funcs else None
            )

    return best_network, best_c, best_psi_funcs, best_d, best_proj_norm


# ---------------------------------------------------------------------------
# Outer Galerkin column construction
# ---------------------------------------------------------------------------

def _build_outer_columns(
    basis_functions: list[BasisFunction],
    X: torch.Tensor,
    u0: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Assemble the outer-Galerkin design matrix.

    Column order: [u_0?, φ_σ,1, (φ_Ψ,1?), φ_σ,2, (φ_Ψ,2?), ...].
    """
    cols: list[torch.Tensor] = []
    if u0 is not None:
        with torch.no_grad():
            cols.append(u0(X))
    for bf in basis_functions:
        cols.append(bf.evaluate_sigma(X))
        if bf.psi_funcs:
            cols.append(bf.evaluate_psi(X))
    if not cols:
        return torch.empty(X.shape[0], 0, device=X.device)
    return torch.column_stack(cols)


# ---------------------------------------------------------------------------
# XGNNModel — stores the fitted model for prediction
# ---------------------------------------------------------------------------

@dataclass
class XGNNModel:
    """Fitted xGNN regression model.

    Prediction:
        u(x) = α_0 u_0(x) + Σ_j (α_σ,j φ_σ,j(x) + α_Ψ,j φ_Ψ,j(x))
    where α_*,* are the entries of `outer_coeffs` in the column order
    produced by `_build_outer_columns`.
    """

    basis_functions: list[BasisFunction]
    outer_coeffs: torch.Tensor
    u0: Optional[Callable[[torch.Tensor], torch.Tensor]]
    device: torch.device

    def predict(self, x_new: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = torch.as_tensor(x_new, dtype=torch.float32, device=self.device)
            if x.ndim == 1:
                x = x.unsqueeze(-1)
            A = _build_outer_columns(self.basis_functions, x, u0=self.u0)
            if A.shape[1] == 0:
                return torch.zeros(x.shape[0], device=self.device)
            return A @ self.outer_coeffs


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def xgnn_regression(
    X: torch.Tensor,
    y: torch.Tensor,
    u0: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    tol: float = 1e-3,
    max_iter: int = 10,
    net_width: int = 32,
    net_depth: int = 2,
    phi_templates: Optional[list[tuple[Callable, float]]] = None,
    lr: float = 1e-3,
    num_epochs: int = 1000,
    num_restarts: int = 3,
    device: Optional[str] = None,
    net_factory: Optional[Callable[[int], nn.Module]] = None,
    ridge: float = 1e-8,
) -> XGNNModel:
    """Run the xGNN regression algorithm.

    Args:
        X: Input data, shape (N,) or (N, d).
        y: Target values, shape (N,).
        u0: Optional initial approximation. Callable mapping (N, d) -> (N,).
            If provided, it is used as the first column of the outer
            Galerkin design matrix and to initialise the residual.
        tol: Stopping tolerance on the a posteriori estimator <r, φ_i>.
        max_iter: Maximum number of basis functions.
        net_width: Width of each sub-network (used when net_factory is None).
        net_depth: Number of hidden layers in each sub-network (used when
            net_factory is None).
        phi_templates: Optional list of (Φ, μ_init) knowledge-based
            functions. Each callable has signature
            ``Phi(X_tensor, mu_param) -> (N,) tensor``.
        lr: Learning rate for Adam (applied to W, b, μ only).
        num_epochs: Training epochs per basis function.
        num_restarts: Number of random restarts per basis function.
        device: Torch device string. Defaults to 'cuda'.
        net_factory: Optional ``(input_dim) -> nn.Module`` returning hidden
            activations of shape (N, width). Defaults to SmallNetwork with
            net_width and net_depth.
        ridge: *Relative* Tikhonov regularisation for the inner / outer
            normal equations (multiplied by the mean diagonal of BᵀB).
            Keeps backward off the SVD path used by lstsq, and stabilises
            rank-deficient B / A. Auto-escalated by 10× up to ten times
            if the solve still reports singular. Default 1e-8.

    Returns:
        Fitted XGNNModel.
    """
    if device is None:
        device = "cuda"
    dev = torch.device(device)

    if net_factory is None:
        _w, _d = net_width, net_depth
        def net_factory(input_dim: int) -> nn.Module:
            return SmallNetwork(input_dim=input_dim, width=_w, depth=_d)

    if phi_templates is None:
        phi_templates = []

    X = torch.as_tensor(X, dtype=torch.float32, device=dev)
    y = torch.as_tensor(y, dtype=torch.float32, device=dev)
    if X.ndim == 1:
        X = X.unsqueeze(-1)

    N, d = X.shape

    # Initial residual r_0 = y - u_0(X)
    if u0 is not None:
        with torch.no_grad():
            residuals = y - u0(X)
    else:
        residuals = y.clone()

    norm_y = torch.linalg.norm(y) + 1e-30

    basis_functions: list[BasisFunction] = []
    outer_coeffs = torch.empty(0, device=dev)

    logger.info(
        "xGNN regression: N=%d, d=%d, tol=%.2e, max_iter=%d, u0=%s, n_phi=%d",
        N, d, tol, max_iter, "yes" if u0 is not None else "no",
        len(phi_templates),
    )

    for i in range(max_iter):
        _factory = lambda: net_factory(d)
        trained_net, trained_c, trained_psi, trained_d, proj_norm = (
            train_basis_function(
                X, residuals, _factory, phi_templates, lr, num_epochs,
                num_restarts=num_restarts, device=dev, ridge=ridge,
            )
        )

        rel_residual = float(torch.linalg.norm(residuals) / norm_y)
        logger.info(
            "  Iter %d: <r, phi> ≈ %.6e, ||r||/||y|| = %.6f",
            i + 1, proj_norm, rel_residual,
        )

        # A posteriori stopping (eq. 2.37)
        if proj_norm < tol:
            logger.info(
                "  Stopping: <r, phi> (%.2e) < tol (%.2e)", proj_norm, tol
            )
            break

        # Freeze the trained basis function
        for p in trained_net.parameters():
            p.requires_grad_(False)
        frozen_psi = []
        for fn, mu in trained_psi:
            mu.requires_grad_(False)
            frozen_psi.append((fn, mu))

        bf = BasisFunction(
            network=trained_net,
            c=trained_c,
            psi_funcs=frozen_psi,
            d=trained_d,
        )
        basis_functions.append(bf)

        # Outer Galerkin: separate coefficients for σ-part and Ψ-part
        A = _build_outer_columns(basis_functions, X, u0=u0)
        outer_coeffs = _ridge_solve(A, y, ridge)

        # Update residual against the refitted approximation
        residuals = y - A @ outer_coeffs

        # Secondary stopping: vanishingly small residual
        rel_res_after = float(torch.linalg.norm(residuals) / norm_y)
        if rel_res_after < tol:
            logger.info(
                "  Stopping: relative residual (%.2e) < tol (%.2e)",
                rel_res_after, tol,
            )
            break

    return XGNNModel(
        basis_functions=basis_functions,
        outer_coeffs=outer_coeffs,
        u0=u0,
        device=dev,
    )
