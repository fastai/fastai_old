from .torch_core import *
from .basic_train import *
from .data import *
from .layers import *

__all__ = ['ConvLearner', 'create_body', 'create_head', 'num_features']

def create_body(model:Model, cut:Optional[int]=None, body_fn:Callable[[Model],Model]=None):
    "Cut off the body of a typically pretrained model at `cut` or as specified by `body_fn`"
    return (nn.Sequential(*list(model.children())[:cut]) if cut
            else body_fn(model) if body_fn else model)

def num_features(m:Model)->int:
    "Return the number of output features for a model"
    for l in reversed(flatten_model(m)):
        if hasattr(l, 'num_features'): return l.num_features

def create_head(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5):
    """Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes.
    :param ps: dropout, can be a single float or a list for each layer"""
    lin_ftrs = [nf, 512, nc] if lin_ftrs is None else [nf] + lin_ftrs + [nc]
    ps = listify(ps)
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    layers = [AdaptiveConcatPool2d(), Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
        layers += bn_drop_lin(ni,no,True,p,actn)
    return nn.Sequential(*layers)

def _default_split(m:Model):
    "By default split models between first and second layer"
    return split_model(m, m[1])

def _resnet_split(m:Model):
    "Split a resnet style model"
    return split_model(m, (m[0][6],m[1]))

_default_meta = {'cut':-1, 'split':_default_split}
_resnet_meta  = {'cut':-2, 'split':_resnet_split }

model_meta = {
    tvm.resnet18 :{**_resnet_meta}, tvm.resnet34: {**_resnet_meta},
    tvm.resnet50 :{**_resnet_meta}, tvm.resnet101:{**_resnet_meta},
    tvm.resnet152:{**_resnet_meta}}

class ConvLearner(Learner):
    "Builds convnet style learners"
    def __init__(self, data:DataBunch, arch:Callable, cut=None, pretrained:bool=True,
                 lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5,
                 custom_head:Optional[nn.Module]=None, split_on:Optional[SplitFuncOrIdxList]=None, **kwargs:Any)->None:
        meta = model_meta.get(arch, _default_meta)
        torch.backends.cudnn.benchmark = True
        body = create_body(arch(pretrained), ifnone(cut,meta['cut']))
        nf = num_features(body) * 2
        head = custom_head or create_head(nf, data.c, lin_ftrs, ps)
        model = nn.Sequential(body, head)
        super().__init__(data, model, **kwargs)
        self.split(ifnone(split_on,meta['split']))
        if pretrained: self.freeze()
        apply_init(model[1], nn.init.kaiming_normal_)
