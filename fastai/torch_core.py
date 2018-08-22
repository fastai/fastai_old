from .imports.torch import *
from .core import *

default_device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

Rank0Tensor = NewType('OneEltTensor', Tensor)
LossFunction = Callable[[Tensor, Tensor], Rank0Tensor]
Metric = Callable[[Tensor, Tensor], float]

def to_device(device:torch.device, b:Collection): return [o.to(device) for o in b]
def to_half(b:Tuple[Tensor, Tensor]):  return [b[0].half(), b[1]]

def split_model(model:nn.Module, idx:Sequence[int]) -> List[nn.Module]:
    "Split the Sequential model according to the layers index in idx"
    layers = list(model.children())
    if idx[0] != 0: idx = [0] + idx
    if idx[-1] != len(layers): idx.append(len(layers))
    return [nn.Sequential(*layers[i:j]) for i,j in zip(idx[:-1],idx[1:])]

