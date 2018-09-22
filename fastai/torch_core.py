from .imports.torch import *
from .core import *

AffineMatrix = Tensor
BoolOrTensor = Union[bool,Tensor]
FloatOrTensor = Union[float,Tensor]
FlowField = Tensor
ItemsList = Collection[Union[Tensor,ItemBase,'ItemsList',float,int]]
LambdaFunc = Callable[[Tensor],Tensor]
LayerFunc = Callable[[nn.Module],None]
Model = nn.Module
ModuleList = Collection[nn.Module]
OptOptimizer = Optional[optim.Optimizer]
ParamList = Collection[nn.Parameter]
Rank0Tensor = NewType('OneEltTensor', Tensor)
SplitFunc = Callable[[Model], List[Model]]
SplitFuncOrIdxList = Union[Callable, Collection[ModuleList]]
TensorOrNumber = Union[Tensor,Number]
TensorOrNumList = Collection[TensorOrNumber]
TensorImage = Tensor
TensorImageSize = Tuple[int,int,int]
Tensors = Union[Tensor, Collection['Tensors']]
Weights = Dict[str,Tensor]

AffineFunc = Callable[[KWArgs], AffineMatrix]
HookFunc = Callable[[Model, Tensors, Tensors], Any]
LogitTensorImage = TensorImage
LossFunction = Callable[[Tensor, Tensor], Rank0Tensor]
MetricFunc = Callable[[Tensor,Tensor],TensorOrNumber]
MetricFuncList = Collection[MetricFunc]
MetricsList = Collection[TensorOrNumber]
OptLossFunc = Optional[LossFunction]
OptMetrics = Optional[MetricsList]
OptSplitFunc = Optional[SplitFunc]
PixelFunc = Callable[[TensorImage, ArgStar, KWArgs], TensorImage]

CoordFunc = Callable[[FlowField, TensorImageSize, ArgStar, KWArgs], LogitTensorImage]
LightingFunc = Callable[[LogitTensorImage, ArgStar, KWArgs], LogitTensorImage]

bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
AdamW = partial(optim.Adam, betas=(0.9,0.99))

def to_data(b:ItemsList):
    "Recursively maps lists of items to their wrapped data"
    if is_listy(b): return [to_data(o) for o in b]
    return b.data if isinstance(b,ItemBase) else b

def to_device(b:Tensors, device:torch.device):
    "Ensure `b` is on `device`"
    device = ifnone(device, default_device)
    if is_listy(b): return [to_device(o, device) for o in b]
    return b.to(device)

def data_collate(batch:ItemsList)->Tensor:
    "Convert `batch` items to tensor data"
    return torch.utils.data.dataloader.default_collate(to_data(batch))

def requires_grad(l:nn.Module, b:Optional[bool]=None)->Optional[bool]:
    "If b is not set requires_grad on all params in l, else return requires_grad of first param"
    ps = list(l.parameters())
    if not ps: return None
    if b is None: return ps[0].requires_grad
    for p in ps: p.requires_grad=b

def trainable_params(m:nn.Module)->ParamList:
    "Return list of trainable params in `m`"
    res = filter(lambda p: p.requires_grad, m.parameters())
    return res

def children(m:nn.Module)->ModuleList:
    "Get children of module"
    return list(m.children())

def num_children(m:nn.Module)->int:
    "Get number of child modules in module"
    return len(children(m))

def range_children(m:nn.Module)->Iterator[int]:
    "Return iterator of len of children of m"
    return range(num_children(m))

flatten_model=lambda l: sum(map(flatten_model,l.children()),[]) if num_children(l) else [l]
def first_layer(m:nn.Module)->nn.Module:
    "Retrieve first layer in a module"
    return flatten_model(m)[0]

def split_model_idx(model:nn.Module, idxs:Collection[int])->ModuleList:
    "Split the model according to the indices in [idxs]"
    layers = flatten_model(model)
    if idxs[0] != 0: idxs = [0] + idxs
    if idxs[-1] != len(layers): idxs.append(len(layers))
    return [nn.Sequential(*layers[i:j]) for i,j in zip(idxs[:-1],idxs[1:])]

def split_model(model:nn.Module, splits:Collection[ModuleList], want_idxs:bool=False):
    "Split the model according to the layers in [splits]"
    layers = flatten_model(model)
    idxs = [layers.index(first_layer(s)) for s in listify(splits)]
    res = split_model_idx(model, idxs)
    return (res,idxs) if want_idxs else res

#TODO: add the test to put bias with bn layers
def split_bn_bias(layer_groups:ModuleList)->ModuleList:
    "Sort each layer in  `layer_groups` into batchnorm (`bn_types`) and non-batchnorm groups"
    split_groups = []
    for l in layer_groups:
        l1,l2 = [],[]
        for c in l.children():
            if isinstance(c, bn_types): l2.append(c)
            else:                       l1.append(c)
        split_groups += [nn.Sequential(*l1), nn.Sequential(*l2)]
    return split_groups

def set_bn_eval(m:nn.Module)->None:
    "Set bn layers in eval mode for all recursive children of m"
    for l in m.children():
        if isinstance(l, bn_types) and not next(l.parameters()).requires_grad:
            l.eval()
        set_bn_eval(l)

def to_half(b:Collection[Tensor])->Collection[Tensor]:
    "[x,y] -> [x.half(),y] (half precision)"
    return [b[0].half(), b[1]]

def bn2float(module:nn.Module)->nn.Module:
    "If a module is batchnorm don't use half precision"
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): module.float()
    for child in module.children(): bn2float(child)
    return module

def model2half(model:nn.Module)->nn.Module:
    "Converts the model to half precision except the batchnorm layers"
    return bn2float(model.half())

def cond_init(m:nn.Module, init_fn:LayerFunc):
    "Initialize the non-batchnorm layers"
    if (not isinstance(m, bn_types)) and requires_grad(m):
        if hasattr(m, 'weight'): init_fn(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'): m.bias.data.fill_(0.)

def apply_leaf(m:nn.Module, f:LayerFunc):
    "Apply `f` to children of m"
    c = children(m)
    if isinstance(m, nn.Module): f(m)
    for l in c: apply_leaf(l,f)

def apply_init(m, init_fn:LayerFunc):
    "Initialize all non-batchnorm layers of model with `init_fn`"
    apply_leaf(m, partial(cond_init, init_fn=init_fn))

def in_channels(m:Model) -> List[int]:
    "Returns the shape of the first weight layer"
    for l in flatten_model(m):
        if hasattr(l, 'weight'): return l.weight.shape[1]
    raise Exception('No weight layer')

def calc_loss(y_pred:Tensor, y_true:Tensor, loss_class:type=nn.CrossEntropyLoss):
    "Calculate loss between `y_pred` and `y_true` using `loss_class`"
    loss_dl = DataLoader(TensorDataset(tensor(y_pred),tensor(y_true)), bs)
    with torch.no_grad():
        return torch.cat([loss_class(reduction='none')(*b) for b in loss_dl])

class DatasetBase(Dataset):
    "Base class for all fastai datasets"
    def __len__(self): return len(self.x)
    @property
    def c(self):
        "Number of classes expressed by dataset y variable"
        return self.y.shape[-1] if len(self.y.shape)>1 else 1
    def __repr__(self): return f'{type(self).__name__} of len {len(self)}'

class LabelDataset(DatasetBase):
    "Base class for fastai datasets that do classification"
    @property
    def c(self):
        "Number of classes expressed by dataset y variable"
        return len(self.classes)