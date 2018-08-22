from ..torch_core import *
from ..callback import *
from ..basic_train import *
from torch._utils import _unflatten_dense_tensors
from torch.nn.utils import parameters_to_vector

__all__ = [MixedPrecision]

def bn2float(module:nn.Module) -> nn.Module:
    "Puts all batchnorm layers back in FP32."
    if isinstance(module, nn.modules.batchnorm._BatchNorm): module.float()
    for child in module.children(): bn2float(child)
    return module

def model2half(model:nn.Module) -> nn.Module:
    "Converts the model to half precision except the batchnorm layers"
    return bn2float(model.half())

def get_master(layer_groups:Collection[nn.Module], flat_master:bool=False) -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
    "Returns two lists, one for the model parameters in FP16 and one for the master parameters in FP32"
    model_params = [[param for param in lg.parameters() if param.requires_grad] for lg in layer_groups]
    if flat_master:
        master_params = [parameters_to_vector([param.data.float() for param in lg]) for lg in model_params]
        master_params = [torch.nn.Parameter(mp, requires_grad=True) for mp in master_params]
        for mp in master_params:
            if mp.grad is None: mp.grad = mp.new(*mp.size())
        return model_params, [[mp] for mp in master_params]
    else:
        master_params = [[param.clone().float().detach() for param in lg] for lg in model_params]
        for mp in master_params:
            for param in mp: param.requires_grad = True
        return model_params, master_params

def model_g2master_g(model_params:Sequence[Sequence[Tensor]], master_params:Sequence[Sequence[Tensor]], flat_master:bool=False):
    "Copies the model gradients to the master parameters for the optimizer step"
    if flat_master:
        for model_group,master_group in zip(model_params,master_params):
            master_group[0].grad.data.copy_(parameters_to_vector([p.grad.data.float() for p in model_group]))
    else:
        for model_group,master_group in zip(model_params,master_params):
            for model, master in zip(model_group, master_group):
                if model.grad is not None:
                    if master.grad is None: master.grad = master.data.new(*master.data.size())
                    master.grad.data.copy_(model.grad.data)
                else: master.grad = None

def master2model(model_params:Sequence[Sequence[Tensor]], master_params:Sequence[Sequence[Tensor]], flat_master:bool=False):
    "Copy master parameters to model parameters"
    if flat_master:
        for model_group,master_group in zip(model_params,master_params):
            for model, master in zip(model_group, _unflatten_dense_tensors(master_group[0].data, model_group)):
                model.data.copy_(master)
    else:
        for model_group,master_group in zip(model_params,master_params):
            for model, master in zip(model_group, master_group): model.data.copy_(master.data)

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
        self.model_params, self.master_params = get_master(self.learn.layer_groups, self.flat_master)
        #Changes the optimizer so that the optimization step is done in FP32.
        opt = self.learn.opt
        mom,wd,beta = opt.mom,opt.wd,opt.beta
        opt_params = [{'params': mp, 'lr': lr} for mp,lr in zip(self.master_params, self.learn.opt._lr)]
        self.learn.opt.opt = self.learn.opt_fn(opt_params)
        opt.mom,opt.wd,opt.beta = mom,wd,beta

    def on_loss_begin(self, last_output:Tensor, **kwargs) -> Tensor:
        #It's better to compute the loss in FP32, to avoid reduction overflow.
        return last_output.float()

    def on_backward_begin(self, last_loss:Rank0Tensor, **kwargs) -> Rank0Tensor:
        #To avoid gradient underflow, we scale the gradients
        return last_loss * self.loss_scale

    def on_backward_end(self, **kwargs):
        #Convert the gradients back to FP32 and divide them by the scale.
        model_g2master_g(self.model_params, self.master_params, self.flat_master)
        for group in self.master_params:
            for param in group: param.grad.div_(self.loss_scale)

    def on_step_end(self, **kwargs):
        #Zeros the gradients of the model since the optimizer is disconnected.
        self.learn.model.zero_grad()
        #Update the params from master to model.
        master2model(self.model_params, self.master_params, self.flat_master)
