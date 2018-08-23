from .data import *
from .torch_core import *

__all__ = [OptimWrapper, Callback, CallbackHandler, Recorder, Stepper]

class OptimWrapper():
    "Basic wrapper around an optimizer to simplify HP changes"
    def __init__(self, opt:optim.Optimizer, wd:Floats=0., true_wd:bool=False):
        self.opt,self.true_wd = opt,true_wd
        self.opt_keys = list(self.opt.param_groups[0].keys())
        self.opt_keys.remove('params')
        self.read_defaults()
        self._wd = self.listify(wd, self.opt.param_groups)

    def __repr__(self) -> str:
        return f'OptimWrapper over {repr(self.opt)}.\nTrue weight decay: {self.true_wd}'

    #Pytorch optimizer methods
    def step(self):
        "Delegate to the optimizer step"
        # weight decay outside of optimizer step (AdamW)
        if self.true_wd:
            for lr,wd,pg in zip(self._lr,self._wd,self.opt.param_groups):
                for p in pg['params']: p.data.mul_(1 - wd*lr)
            self.set_val('weight_decay', self.listify(0, self._wd))
        self.opt.step()

    def zero_grad(self): self.opt.zero_grad()

    #Hyperparameters as properties
    @property
    def lr(self) -> float: return self._lr[-1]

    @lr.setter
    def lr(self, val:Floats): self._lr = self.set_val('lr', self.listify(val, self._lr))

    @property
    def mom(self) -> float: return self._mom[-1]

    @mom.setter
    def mom(self, val:Floats):
        if 'momentum' in self.opt_keys: self.set_val('momentum', self.listify(val, self._mom))
        elif 'betas' in self.opt_keys:  self.set_val('betas', (self.listify(val, self._mom), self._beta))
        self._mom = self.listify(val, self._mom)

    @property
    def beta(self) -> float: return None if self._beta is None else self._beta[-1]

    @beta.setter
    def beta(self, val:Floats):
        if val is None: return
        if 'betas' in self.opt_keys:    self.set_val('betas', (self._mom, self.listify(val, self._beta)))
        elif 'alpha' in self.opt_keys:  self.set_val('alpha', self.listify(val, self._beta))
        self._beta = self.listify(val, self._beta)

    @property
    def wd(self) -> float: return self._wd[-1]

    @wd.setter
    def wd(self, val:Floats):
        if not self.true_wd: self.set_val('weight_decay', self.listify(val, self._wd))
        self._wd = self.listify(val, self._wd)

    #Helper functions
    def read_defaults(self):
        "Read the values inside the optimizer for the hyper-parameters"
        self._beta = None
        if 'lr' in self.opt_keys: self._lr = self.read_val('lr')
        if 'momentum' in self.opt_keys: self._mom = self.read_val('momentum')
        if 'alpha' in self.opt_keys: self._beta = self.read_val('alpha')
        if 'betas' in self.opt_keys: self._mom,self._beta = self.read_val('betas')
        if 'weight_decay' in self.opt_keys: self._wd = self.read_val('weight_decay')

    def set_val(self, key:str, val:Union[Floats, Tuple[Floats, Floats]]):
        "Set the values inside the optimizer dictionary at the key"
        if is_tuple(val): val = [(v1,v2) for v1,v2 in zip(*val)]
        for v,pg in zip(val,self.opt.param_groups): pg[key] = v
        return val

    def read_val(self, key:str) -> Union[Floats, Tuple[Floats, Floats]]:
        "Read a hyper-parameter key in the optimizer dictionary."
        val = [pg[key] for pg in self.opt.param_groups]
        if is_tuple(val[0]): val = [o[0] for o in val], [o[1] for o in val]
        return val

    def listify(self, p, q) -> List[Any]:
        "Wrap listify with an assert."
        if is_listy(p): assert len(p) == len(q), f'Passing {len(p)} hyperparameters when we have {len(q)} groups.'
        return listify(p,q)

class Callback():
    "Basic definition of a callback"

    def on_train_begin(self, **kwargs): pass
        #To initiliaze constants in the callback.
    def on_epoch_begin(self, **kwargs): pass
        #At the beginning of each epoch
    def on_batch_begin(self, **kwargs) -> Tuple[Tensor, Tensor]: pass
        #To set HP before the step is done.
        #Returns xb, yb (which can allow us to modify the input at that step if needed)
    def on_loss_begin(self, **kwargs) -> Tensor: pass
        #Called after the forward pass but before the loss has been computed.
        #Returns the output (which can allow us to modify it)
    def on_backward_begin(self, **kwargs) -> Rank0Tensor: pass
        #Called after the forward pass and the loss has been computed, but before the back propagation.
        #Returns the loss (which can allow us to modify it, for instance for reg functions)
    def on_backward_end(self, **kwargs): pass
        #Called after the back propagation had been done (and the gradients computed) but before the step of the optimizer.
        #Useful for true weight decay in AdamW
    def on_step_end(self, **kwargs): pass
        #Called after the step of the optimizer but before the gradients are zeroed (not sure this one is useful)
    def on_batch_end(self, **kwargs) -> bool: pass
        #Called at the end of the batch
    def on_epoch_end(self, **kwargs) -> bool: pass
        #Called at the end of an epoch
    def on_train_end(self, **kwargs): pass
        #Useful for cleaning up things and saving files/models

@dataclass
class CallbackHandler():
    "Class actually called by the training loop and responsible for dispatching to callbacks"

    callbacks:Collection[Callback]
    beta:float = 0.98

    def __post_init__(self):
        self.state_dict:Dict[str,Union[int,float,Tensor]] = {'epoch': 0, 'iteration': 0, 'num_batch': 0}
        self.smoothener = SmoothenValue(self.beta)

    def __call__(self, cb_name:str):
        return [getattr(cb, f'on_{cb_name}')(**self.state_dict) for cb in self.callbacks]

    def __repr__(self) -> str:
        res = ''
        for cb in self.callbacks: res += repr(cb)
        return res

    def on_train_begin(self):
        self.state_dict:Dict[str,Union[int,float,Tensor]] = {'epoch': 0, 'iteration': 0, 'num_batch': 0}
        self('train_begin')

    def on_epoch_begin(self):
        self.state_dict['num_batch'] = 0
        self('epoch_begin')

    def on_batch_begin(self, xb:Tensor, yb:Tensor) -> Tuple[Tensor, Tensor]:
        self.state_dict['last_input'], self.state_dict['last_target'] = xb, yb
        for cb in self.callbacks:
            a = cb.on_batch_begin(**self.state_dict)
            if a is not None: self.state_dict['last_input'], self.state_dict['last_target'] = a
        return self.state_dict['last_input'], self.state_dict['last_target']

    def on_loss_begin(self, out:Tensor) -> Tensor:
        self.state_dict['last_output'] = out
        for cb in self.callbacks:
            a = cb.on_loss_begin(**self.state_dict)
            if a is not None: self.state_dict['last_output'] = a
        return self.state_dict['last_output']

    def on_backward_begin(self, loss) -> Rank0Tensor:
        self.smoothener.add_value(loss.detach())
        self.state_dict['last_loss'], self.state_dict['smooth_loss'] = loss, self.smoothener.smooth
        for cb in self.callbacks:
            a = cb.on_backward_begin(**self.state_dict)
            if a is not None: self.state_dict['last_loss'] = a
        return self.state_dict['last_loss']

    def on_backward_end(self):        self('backward_end')
    def on_step_end(self):            self('step_end')

    def on_batch_end(self, loss) -> bool:
        self.state_dict['last_loss'] = loss
        stop = np.any(self('batch_end'))
        self.state_dict['iteration'] += 1
        self.state_dict['num_batch'] += 1
        return stop

    def on_epoch_end(self, val_metrics) -> bool:
        self.state_dict['last_metrics'] = val_metrics
        stop = np.any(self('epoch_end'))
        self.state_dict['epoch'] += 1
        return stop

    def on_train_end(self): self('train_end')

@dataclass
class Recorder(Callback):
    "Callback present in every learner to record basic stats"

    opt:OptimWrapper
    nb_epoch:int
    train_dl:DeviceDataLoader=None
    pbar:Callable=None

    def __repr__(self) -> str:
        return f'Recorder\n  internal optimizer:{repr(self.opt)}\n  internal dataloader:{repr(self.train_dl)}'

    def on_train_begin(self, **kwargs):
        self.losses,self.val_losses,self.lrs,self.moms,self.metrics,self.nb_batches = [],[],[],[],[],[]

    def on_batch_begin(self, **kwargs):
        self.lrs.append(self.opt.lr)
        self.moms.append(self.opt.mom)

    def on_backward_begin(self, smooth_loss:float, **kwargs):
        #We record the loss here before any other callback has a chance to modify it.
        self.losses.append(smooth_loss)
        if self.pbar is not None and hasattr(self.pbar,'child'):
            self.pbar.child.comment = f'{smooth_loss:.4f}'

    def on_epoch_end(self, epoch:int, num_batch:int, smooth_loss:float, last_metrics:Sequence[float], **kwargs):
        self.nb_batches.append(num_batch)
        if last_metrics is not None:
            self.val_losses.append(last_metrics[0])
            if len(last_metrics) > 1: self.metrics.append(last_metrics[1:])
            self.pbar.write(f'{epoch}, {smooth_loss}, {*last_metrics}')
            self.pbar.update_graph(*self.send_graphs())
        else:  self.pbar.write(f'{epoch}, {smooth_loss}')

    def plot_lr(self, show_moms:bool=False):
        iterations = list(range(len(self.lrs)))
        if show_moms:
            _, axs = plt.subplots(1,2, figsize=(12,4))
            axs[0].plot(iterations, self.lrs)
            axs[1].plot(iterations, self.moms)
        else: plt.plot(iterations, self.lrs)

    def plot(self, skip_start:int=10, skip_end:int=5):
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]
        _, ax = plt.subplots(1,1)
        ax.plot(lrs, losses)
        ax.set_xscale('log')

    def plot_losses(self):
        _, ax = plt.subplots(1,1)
        iterations = list(range(len(self.losses)))
        ax.plot(iterations, self.losses)
        val_iter = self.nb_batches
        val_iter = np.cumsum(val_iter)
        ax.plot(val_iter, self.val_losses)

    def plot_metrics(self):
        assert len(self.metrics) != 0, "There is no metrics to plot."
        _, axes = plt.subplots(len(self.metrics[0]),1,figsize=(6, 4*len(self.metrics[0])))
        val_iter = self.nb_batches
        val_iter = np.cumsum(val_iter)
        axes = axes.flatten() if len(self.metrics[0]) != 1 else [axes]
        for i, ax in enumerate(axes):
            values = [met[i] for met in self.metrics]
            ax.plot(val_iter, values)

    def send_graphs(self):
        iters = list(range(len(self.losses)))
        val_iter = np.array(self.nb_batches).cumsum()
        x_bounds = (0, (self.nb_epoch - len(self.nb_batches)) * self.nb_batches[-1] + len(self.losses))
        y_bounds = (0, max((max(self.losses), max(self.val_losses))))
        return [(iters, self.losses), (val_iter, self.val_losses)], x_bounds, y_bounds

class Stepper():
    def __init__(self, vals:Union[float,Tuple[float,float]], num_it:int, ft:AnnealingFt=None):
        self.start,self.end = (vals[0],vals[1]) if is_tuple(vals) else (vals,0)
        self.num_it = num_it
        if ft is None: self.ft = annealing_linear if is_tuple(vals) else annealing_no
        else:          self.ft = ft
        self.n = 0

    def step(self) -> float:
        self.n += 1
        return self.ft(self.start, self.end, self.n/self.num_it)

    def repr(self) -> str:
        return f'Stepper from {self.start} to {self.end} with {self.ft.__name__}. Current step: {self.n/self.num_it}'

    @property
    def is_done(self) -> bool:  return self.n >= self.num_it
