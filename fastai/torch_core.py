from .imports.core import *
from .imports.torch import *

default_device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

Rank0Tensor = NewType('OneEltTensor', Tensor)
LossFunction = Callable[[Tensor, Tensor], Rank0Tensor]
Metric = Callable[[Tensor, Tensor], float]

def to_device(device:torch.device, b:Collection): return [o.to(device) for o in b]
def to_half(b:Tuple[Tensor, Tensor]):  return [b[0].half(), b[1]]