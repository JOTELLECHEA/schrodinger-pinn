import torch

def infinite_potential_well(x: torch.Tensor, L: float = 1.0)-> torch.Tensor:
    '''V(x) = 0 inside [-L, L]'''
    return torch.zeros_like(x)
