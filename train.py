import torch
import math
from tqdm import tqdm
from pathlib import Path
from src.models import QuantumPINN
from src.potentials import infinite_potential_well
from src.losses.tise_1d import pde_residual_loss, boundary_loss, normalization_loss

def main():
    torch.manual_seed(137
                    )
    L = 1.0
    n_collocation = 200
    adam_epochs = 5000
    lbfgs_epochs = 500
    
    w_pde = 1.0
    w_bc = 10.0
    w_norm = 10.0


    x_collocation = torch.linspace(-L, L, n_collocation).view(-1, 1)
    
    model = QuantumPINN(hidden_dim=50, depth=4, e_init=1.0)
    print(f"Initial Energy Guess E: {model.E.item():.4f}\n")

    def compute_total_loss():
        l_pde = pde_residual_loss(model, x_collocation, infinite_potential_well)
        l_bc = boundary_loss(model, L=L)
        l_norm = normalization_loss(model, x_collocation, L=L)
        return w_pde * l_pde + w_bc * l_bc + w_norm * l_norm

    # Adam Optimizer
    print(f"{' Starting Phase 1: Adam Optimizer ':-^50}")
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)

    pbar = tqdm(range(1, adam_epochs + 1), desc="Adam")

    for _ in pbar:
        optimizer_adam.zero_grad()
        loss = compute_total_loss()
        loss.backward()
        optimizer_adam.step()

        pbar.set_postfix({
        "loss": f"{loss.item():.3e}",
        "E": f"{model.E.item():.6f}",
        })


    # L-BFGS Optimizer
    print()
    print(f"{' Starting Phase 2: L-BFGS Optimizer ':-^50}") 
    
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=50,
        line_search_fn="strong_wolfe"
    )
    pbar = tqdm(range(1, lbfgs_epochs + 1), desc="L-BFGS")

    for _ in pbar:
        
        def closure():
            optimizer_lbfgs.zero_grad()
            loss = compute_total_loss()
            loss.backward()
            return loss

        optimizer_lbfgs.step(closure)

        loss = optimizer_lbfgs.step(closure)
        pbar.set_postfix({
        "loss": f"{loss.item():.3e}",
        "E": f"{model.E.item():.6f}"
        })
    print()

    # Final Benchmark Check
    exact_E = (math.pi ** 2) / 8.0
    learned_E = model.E.item()
    abs_err = abs(learned_E - exact_E)
    rel_err = (abs_err / exact_E) * 100

    print(f"{' Training Complete ':-^50}") 
    print(f"Exact Ground State E:    {exact_E:.6f}")
    print(f"Discovered Ground State: {learned_E:.6f}")
    print(f"Absolute Error:          {abs_err:.6f}")
    print(f"Relative Error:          {rel_err:.4f}%")
    print(f"{'':-^50}") 
    print()

    # Save Checkpoint
    ckpt = Path("outputs/checkpoints/pinn_tise.pt")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "E": model.E.item(),
                "config": {"L": L, "n_collocation": n_collocation}}, ckpt)
    print("Model checkpoint saved to outputs/checkpoints/pinn_tise.pt")

if __name__ == "__main__":
    main()