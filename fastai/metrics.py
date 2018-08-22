from .imports.torch import *

__all__ = [accuracy]

def accuracy(out:Tensor, yb:Tensor) -> float:
    preds = torch.max(out, dim=1)[1]
    return (preds==yb).float().mean().item()

