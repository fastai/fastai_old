from .imports.core import *
from ipykernel.kernelapp import IPKernelApp

def in_notebook(): return IPKernelApp.initialized()

def to_device(device, b): return [o.to(device) for o in b]
def to_half(b):           return [b[0].half(), b[1]]

def is_tuple(x):    return isinstance(x, tuple)
def is_iterable(x): return isinstance(x, Iterable)

def listify(p=None, q=None):
    if p is None: p=[]
    elif is_iterable(p): p=[p]
    n = q if type(q)==int else 1 if q is None else len(q)
    if len(p)==1: p = p * n
    return p

if in_notebook():  tqdm, trange = tqdm_notebook, tnrange

class SmoothenValue():
    def __init__(self, beta):
        self.beta,self.n,self.mov_avg = beta,0,0
    
    def add_value(self, val):
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)

def annealing_no(start, end, pct): return start
def annealing_linear(start, end, pct): return start + pct * (end-start)
def annealing_exp(start, end, pct): return start * (end/start) ** pct
def annealing_cos(start, end, pct):
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start-end)/2 * cos_out
    
def do_annealing_poly(start, end, pct, degree): return end + (start-end) * (1-pct)**degree
def annealing_poly(degree): return partial(do_annealing_poly, degree=degree)