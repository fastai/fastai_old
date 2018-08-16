from .imports.core import *
from .imports.torch import *
from .basic_train import Learner
from .imports import callbacks as cb
from .callbacks.fp16 import model2half

def lr_find(learn:Learner, start_lr:float=1e-5, end_lr:float=10., num_it:int=100):
    "Lanches the LR Range test"
    #TODO: add model.save and model.load.
    learn.create_opt(start_lr)
    cbs = [cb.LRFinder(learn.opt, learn.data, start_lr, end_lr, num_it)]
    a = int(np.ceil(num_it/len(learn.data.train_dl)))
    learn.fit(a, start_lr, callbacks=cbs)

def fit_one_cyle(learn:Learner, max_lr:float, cyc_len:int, moms:Tuple[float,float]=(0.95,0.85), div_factor:float=10., pct_end:float=0.1,
                 wd:float=0.):
    "Fits a model following the 1cycle policy"
    cbs = [cb.OneCycleScheduler(learn, max_lr, cyc_len, moms, div_factor, pct_end)]
    learn.fit(cyc_len, max_lr/div_factor, wd=wd, callbacks=cbs)

def to_fp16(learn:Learner, loss_scale:float=512., flat_master:bool=False):
    "Tranforms the learner in FP16 precision"
    learn.model = model2half(learn.model)
    learn.callbacks.append(cb.MixedPrecision(learn, loss_scale=loss_scale, flat_master=flat_master))