from .imports.torch import *

def accuracy(out:Tensor, yb:Tensor) -> float:
    preds = torch.max(out, dim=1)[1]
    return (preds==yb).float().mean().item()