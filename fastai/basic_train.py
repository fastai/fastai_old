from .torch_core import *
from .data import *
from .callback import *

__all__ = [loss_batch, fit, Learner]

def loss_batch(model:nn.Module, xb:Tensor, yb:Tensor, loss_fn:LossFunction, opt:OptimWrapper=None,
               cb_handler:CallbackHandler=None, metrics:Collection[Metric]=None) -> Sequence[Union[float,int]]:
    "Computes the loss for a batch and does the corresponding training step (if applicable)"
    if cb_handler is None: cb_handler = CallbackHandler([])
    out = model(xb)
    out = cb_handler.on_loss_begin(out)
    loss = loss_fn(out, yb)
    mets = [f(out,yb) for f in metrics] if metrics is not None else []

    if opt is not None:
        loss = cb_handler.on_backward_begin(loss)
        loss.backward()
        cb_handler.on_backward_end()
        opt.step()
        cb_handler.on_step_end()
        opt.zero_grad()

    return (loss.detach(),) + tuple(mets) + (len(xb),)

def fit(epochs:int, model:nn.Module, loss_fn:LossFunction, opt:OptimWrapper, data:DataBunch,
        callbacks:Collection[Callback]=None, metrics:Collection[Metric]=None, pbar:Callable=None):
    "Training loop of a model"
    cb_handler = CallbackHandler(callbacks)
    cb_handler.on_train_begin()
    if pbar is None: pbar = master_bar(range(epochs))

    for _ in pbar:
        model.train()
        cb_handler.on_epoch_begin()

        for xb,yb in progress_bar(data.train_dl, parent=pbar):
            xb, yb = cb_handler.on_batch_begin(xb, yb)
            loss,_ = loss_batch(model, xb, yb, loss_fn, opt, cb_handler)
            if cb_handler.on_batch_end(loss): break

        if hasattr(data,'valid_dl') and data.valid_dl is not None:
            model.eval()
            with torch.no_grad():
                *val_metrics,nums = zip(*[loss_batch(model, xb, yb, loss_fn, cb_handler=cb_handler, metrics=metrics)
                                for xb,yb in progress_bar(data.valid_dl, parent=pbar)])
            val_metrics = [np.sum(np.multiply(val,nums)) / np.sum(nums) for val in val_metrics]

        else: val_metrics=None
        if cb_handler.on_epoch_end(val_metrics): break

    cb_handler.on_train_end()


@dataclass
class Learner():
    "Object that wraps together some data, a model, a loss function and an optimizer"

    data:DataBunch
    model:nn.Module
    opt_fn:Callable=optim.SGD
    loss_fn:LossFunction=F.cross_entropy
    metrics:Collection[Metric]=None
    true_wd:bool=False
    layer_groups:Collection[nn.Module]=None
    def __post_init__(self):
        self.model = self.model.to(self.data.device)
        self.callbacks = []

    def fit(self, epochs:int, lr:Floats, wd:Floats=0., callbacks:Collection[Callback]=None):
        if not hasattr(self, 'opt'): self.create_opt(lr, wd)
        else: self.opt.wd = wd
        if callbacks is None: callbacks = []
        pbar = master_bar(range(epochs))
        self.recorder = Recorder(self.opt, epochs, self.data.train_dl, pbar)
        callbacks = [self.recorder] + self.callbacks + callbacks
        fit(epochs, self.model, self.loss_fn, self.opt, self.data, callbacks=callbacks, metrics=self.metrics, pbar=pbar)

    def create_opt(self, lr:Floats, wd:Floats=0.):
        if self.layer_groups is None: self.layer_groups = [self.model]
        lrs = listify(lr, self.layer_groups)
        opt = self.opt_fn([{'params':l.parameters(), 'lr':lr} for l,lr in zip(self.layer_groups, lrs)])
        self.opt = OptimWrapper(opt, wd=wd, true_wd=self.true_wd)

