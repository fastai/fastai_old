from ..imports.core import *
from .. import core as c
from ..callback import Callback, Stepper

class LRFinder(Callback):
    
    def __init__(self, opt, data, start_lr=1e-5, end_lr=10, num_it=200):
        self.opt,self.data = opt,data
        self.sched = Stepper((start_lr, end_lr), num_it, c.annealing_exp)
        #To avoid validating if the train_dl has less than num_it batches, we put aside the valid_dl and remove it
        #during the call to fit.
        self.valid_dl = data.valid_dl
        self.data.valid_dl = None
    
    def on_train_begin(self, **kwargs):
        self.opt.lr = self.sched.start
        self.stop,self.best_loss = False,0.
    
    def on_batch_end(self, iteration, smooth_loss, **kwargs):
        if iteration==0 or smooth_loss < self.best_loss: self.best_loss = smooth_loss
        self.opt.lr = self.sched.step()
        if self.sched.is_done or smooth_loss > 4*self.best_loss:
            #We use the smoothed loss to decide on the stopping since it's less shaky.
            self.stop=True
            return True
    
    def on_epoch_end(self, **kwargs): return self.stop
    
    def on_train_end(self, **kwargs):
        #Clean up and put back the valid_dl in its place.
        self.data.valid_dl = self.valid_dl