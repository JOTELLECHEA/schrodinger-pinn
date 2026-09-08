import torch
from typing import Callable

def pde_residual_loss(
        model: torch.nn.Module,
        x_collocation: torch.Tensor,
        potential_fn: Callable[[torch.Tensor], torch.Tensor]
)-> torch.Tensor:
    """Computes Mean Squared Error of the TISE residual: -0.5 * d^2(psi)/dx^2 + V(x)psi - E*psi"""
    x = x_collocation.clone().detach().requires_grad_(True)
    psi = model(x)

    # 1st derivative
    dpsi_dx = torch.autograd.grad(
        psi, x, 
        grad_outputs=torch.ones_like(psi), 
        create_graph=True
    )[0]
    
    # 2nd derivative
    d2psi_dx2 = torch.autograd.grad(
        dpsi_dx, x, 
        grad_outputs=torch.ones_like(dpsi_dx), 
        create_graph=True
    )[0]

    V = potential_fn(x)
    residual = -0.5 * d2psi_dx2 + V * psi - model.E * psi

    return torch.mean(residual **2)

def boundary_loss(model: torch.nn.Module, L: float = 1.0)-> torch.Tensor:
    """Enforces Dirichlet BCs: psi(-L) = 0 and psi(L) = 0"""
    x_bc = torch.tensor([[-L], [L]], dtype=torch.float32)
    psi_bc = model(x_bc)

    return torch.mean(psi_bc ** 2) 

def normalization_loss(model: torch.nn.Module, x_collocation: torch.Tensor, L: float = 1.0) -> torch.Tensor:
    """Enforces integral(|psi|^2 dx) = 1 via trapezoidal quadrature."""
    psi = model(x_collocation)
    domain_length = 2.0 * L
    dx = domain_length / (x_collocation.shape[0] - 1)
    prob_density_integral = torch.trapezoid(psi.squeeze() ** 2, dx=dx)

    return (prob_density_integral - 1.0) ** 2