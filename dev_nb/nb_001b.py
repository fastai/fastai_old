import pickle, gzip, torch, math, numpy as np, torch.nn.functional as F
from pathlib import Path
from IPython.core.debugger import set_trace
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from collections import Iterable
from functools import reduce,partial
from tqdm import tqdm, tqdm_notebook, trange, tnrange

def loss_batch(model, xb, yb, loss_fn, opt=None):
    loss = loss_fn(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_fn, opt, train_dl, valid_dl):
    for epoch in tnrange(epochs):
        model.train()
        it = tqdm_notebook(train_dl, leave=False)
        for xb,yb in it:
            loss,_ = loss_batch(model, xb, yb, loss_fn, opt)
            it.set_postfix_str(loss)

        model.eval()
        with torch.no_grad():
            losses,nums = zip(*[loss_batch(model, xb, yb, loss_fn)
                                for xb,yb in valid_dl])
        val_loss = np.sum(np.multiply(losses,nums)) / np.sum(nums)

        print(epoch, val_loss)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func=func

    def forward(self, x): return self.func(x)

def ResizeBatch(*size): return Lambda(lambda x: x.view((-1,)+size))
def Flatten(): return Lambda(lambda x: x.view((x.size(0), -1)))
def PoolFlatten(): return nn.Sequential(nn.AdaptiveAvgPool2d(1), Flatten())


def is_listy(x): return isinstance(x, (list,tuple))

def listify(x=None, y=None):
    if x is None: x=[]
    elif not is_listy(x): x=[x]
    n = y if type(y)==int else 1 if y is None else len(y)
    if len(x)==1: x = x * n
    return x

def compose(funcs):
    funcs = reversed(listify(funcs))
    return reduce(lambda f, g: lambda x: f(g(x)), funcs, lambda x: x)

class IterPipe():
    def __init__(self, iterator, funcs): self.iter,self.func = iterator,compose(funcs)
    def __len__(self): return len(self.iter)
    def __iter__(self): return map(self.func, self.iter)


def get_dl(ds, bs, shuffle, tfms=None):
    return IterPipe(DataLoader(ds, batch_size=bs, shuffle=shuffle), tfms)


def conv2_relu(nif, nof, ks, stride):
    return nn.Sequential(nn.Conv2d(nif, nof, ks, stride, padding=ks//2), nn.ReLU())

def simple_cnn(actns, kernel_szs, strides):
    layers = [conv2_relu(actns[i], actns[i+1], kernel_szs[i], stride=strides[i])
        for i in range(len(strides))]
    layers.append(PoolFlatten())
    return nn.Sequential(*layers)


def to_device(device, b): return [o.to(device) for o in b]

default_device = torch.device('cuda')


class DataBunch():
    def __init__(self, train_ds, valid_ds, bs=64, device=None, train_tfms=None, valid_tfms=None):
        self.device = default_device if device is None else device
        dev_tfm = [partial(to_device, self.device)]
        self.train_dl = get_dl(train_ds, bs,   shuffle=True,  tfms=dev_tfm + listify(train_tfms))
        self.valid_dl = get_dl(valid_ds, bs*2, shuffle=False, tfms=dev_tfm + listify(valid_tfms))

class Learner():
    def __init__(self, data, model):
        self.data,self.model = data,model.to(data.device)

    def fit(self, epochs, lr, opt_fn=optim.SGD):
        opt = opt_fn(self.model.parameters(), lr=lr)
        loss_fn = F.cross_entropy
        fit(epochs, self.model, loss_fn, opt, self.data.train_dl, self.data.valid_dl)

