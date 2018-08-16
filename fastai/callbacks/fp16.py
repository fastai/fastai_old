from ..imports.core import *
from ..imports.torch import *
from .. import torch_core as tc
from ..callback import Callback
from ..basic_train import Learner
from torch._utils import _unflatten_dense_tensors
from torch.nn.utils import parameters_to_vector

def bn2float(module:nn.Module) -> nn.Module:
    "Puts all batchnorm layers back in FP32."
    if isinstance(module, nn.modules.batchnorm._BatchNorm): module.float()
    for child in module.children(): bn2float(child)
    return module

def model2half(model:nn.Module) -> nn.Module:
    "Converts the model to half precision except the batchnorm layers"
    return bn2float(model.half())

def get_master(model:nn.Module, flat_master:bool=False) -> Tuple[List[Tensor], List[Tensor]]:
    "Returns two lists, one for the model parameters in FP16 and one for the master parameters in FP32"
    model_params = [param for param in model.parameters() if param.requires_grad]
    if flat_master:
        master_params = parameters_to_vector([param.data.float() for param in model_params])
        master_params = torch.nn.Parameter(master_params)
        master_params.requires_grad = True
        if master_params.grad is None: master_params.grad = master_params.new(*master_params.size())
        return model_params, [master_params]
    else:
        master_params = [param.clone().float().detach() for param in model_params]
        for param in master_params: param.requires_grad = True
        return model_params, master_params

def model_g2master_g(model_params:List[Tensor], master_params:List[Tensor], flat_master:bool=False):
    "Copies the model gradients to the master parameters for the optimizer step"
    if flat_master:
        master_params[0].grad.data.copy_(parameters_to_vector([p.grad.data.float() for p in model_params]))
    else:
        for model, master in zip(model_params, master_params):
            if model.grad is not None:
                if master.grad is None: master.grad = master.data.new(*master.data.size())
                master.grad.data.copy_(model.grad.data)
            else: master.grad = None

def master2model(model_params:List[Tensor], master_params:List[Tensor], flat_master:bool=False):
    "Copy master parameters to model parameters"
    if flat_master:
        for model, master in zip(model_params, _unflatten_dense_tensors(master_params[0].data, model_params)):
            model.data.copy_(master)
    else: 
        for model, master in zip(model_params, master_params): model.data.copy_(master.data)

@dataclass
class MixedPrecision(Callback):
    "Callback that handles mixed-precision training"
    learn:Learner
    loss_scale:float=512.
    flat_master:bool=False
    def __post_init__(self): assert torch.backends.cudnn.enabled, "Mixed precision training requires cudnn." 
    
    def on_train_begin(self, **kwargs):
        #Insures the dataloaders are in half precision.
        self.learn.data.train_dl.half = True
        if hasattr(self.learn.data, 'valid_dl') and self.learn.data.valid_dl is not None:
            self.learn.data.valid_dl.half = True
        #Get a copy of the model params in FP32
        self.model_params, self.master_params = get_master(self.learn.model, self.flat_master)
        #Changes the optimizer so that the optimization step is done in FP32.
        opt = self.learn.opt
        mom,wd,beta = opt.mom,opt.wd,opt.beta
        self.learn.opt.opt = self.learn.opt_fn(self.master_params, self.learn.opt.lr)
        opt.mom,opt.wd,opt.beta = mom,wd,beta
    
    def on_loss_begin(self, last_output:Tensor, **kwargs) -> Tensor:
        #It's better to compute the loss in FP32, to avoid reduction overflow.
        return last_output.float()
    
    def on_backward_begin(self, last_loss:tc.Rank0Tensor, **kwargs) -> tc.Rank0Tensor:
        #To avoid gradient underflow, we scale the gradients
        return last_loss * self.loss_scale
    
    def on_backward_end(self, **kwargs):
        #Convert the gradients back to FP32 and divide them by the scale.
        model_g2master_g(self.model_params, self.master_params, self.flat_master)
        for param in self.master_params: param.grad.div_(self.loss_scale)
    
    def on_step_end(self, **kwargs):
        #Zeros the gradients of the model since the optimizer is disconnected.
        self.learn.model.zero_grad()
        #Update the params from master to model.
        master2model(self.model_params, self.master_params, self.flat_master)