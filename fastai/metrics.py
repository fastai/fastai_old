from .imports.core import *
from .imports.torch import *

def accuracy(out, yb):
    preds = torch.max(out, dim=1)[1]
    return (preds==yb).float().mean()