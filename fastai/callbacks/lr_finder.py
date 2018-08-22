from .core import *
from ..data import DataBunch
from ..callback import *

class LRFinder(Callback):
    "Callback that handles the LR range test"

    def __init__(self, opt:OptimWrapper, data:DataBunch, start_lr:float=1e-5, end_lr:float=10, num_it:int=100):
        self.opt,self.data = opt,data
        self.sched = Stepper((start_lr, end_lr), num_it, annealing_exp)
        #To avoid validating if the train_dl has less than num_it batches, we put aside the valid_dl and remove it
        #during the call to fit.
        self.valid_dl = data.valid_dl
        self.data.valid_dl = None
    
    def __repr__(self) -> str:
        return f'LRFinder\nOptim wrapper on: {repr(self.opt)}\nData: {repr(self.data)}'
    
    def on_train_begin(self, **kwargs):
        self.opt.lr = self.sched.start
        self.stop,self.best_loss = False,0.
    
    def on_batch_end(self, iteration:int, smooth_loss:float, **kwargs) -> bool:
        if iteration==0 or smooth_loss < self.best_loss: self.best_loss = smooth_loss
        self.opt.lr = self.sched.step()
        if self.sched.is_done or smooth_loss > 4*self.best_loss:
            #We use the smoothed loss to decide on the stopping since it's less shaky.
            self.stop=True
            return True
    
    def on_epoch_end(self, **kwargs) -> bool: return self.stop
    
    def on_train_end(self, **kwargs):
        #Clean up and put back the valid_dl in its place.
        self.data.valid_dl = self.valid_dl

