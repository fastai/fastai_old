from nb_001b import *
import sys, PIL, matplotlib.pyplot as plt, itertools, math, random, collections
import scipy.stats, scipy.special

from enum import Enum, IntEnum
from torch.utils.data import Dataset
from torch import tensor, FloatTensor, LongTensor, ByteTensor, DoubleTensor, HalfTensor, ShortTensor
from operator import itemgetter, attrgetter
from numpy import cos, sin, tan, tanh, log, exp
from collections import defaultdict, abc, namedtuple
from PIL import Image

def find_classes(folder):
    classes = [d for d in folder.iterdir()
               if d.is_dir() and not d.name.startswith('.')]
    assert(len(classes)>0)
    return sorted(classes, key=lambda d: d.name)

def get_image_files(c):
    return [o for o in list(c.iterdir())
            if not o.name.startswith('.') and not o.is_dir()]

def pil2tensor(image):
    arr = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    arr = arr.view(image.size[1], image.size[0], -1)
    arr = arr.permute(2,0,1)
    return arr.float().div_(255)

class FilesDataset(Dataset):
    def __init__(self, folder, classes=None):
        self.fns, self.y = [], []
        if classes is None: classes = [cls.name for cls in find_classes(folder)]
        self.classes = classes
        for i, cls in enumerate(classes):
            fnames = get_image_files(folder/cls)
            self.fns += fnames
            self.y += [i] * len(fnames)

    def __len__(self): return len(self.fns)

    def __getitem__(self,i):
        x = PIL.Image.open(self.fns[i]).convert('RGB')
        return pil2tensor(x),self.y[i]

def image2np(image): return image.cpu().permute(1,2,0).numpy()

def show_image(img, ax=None, figsize=(3,3), hide_axis=True):
    if ax is None: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(image2np(img))
    if hide_axis: ax.axis('off')

def show_image_batch(dl, classes, rows=None):
    if rows is None: rows = int(math.sqrt(len(x)))
    x,y = next(iter(dl))[:rows*rows]
    show_images(x,y,rows, classes)

def show_images(x,y,rows, classes, figsize=(12,15)):
    fig, axs = plt.subplots(rows,rows,figsize=figsize)
    for i, ax in enumerate(axs.flatten()):
        show_image(x[i], ax)
        ax.set_title(classes[y[i]])
    plt.tight_layout()

def get_batch_stats(dl):
    x,_ = next(iter(dl))
    # hack for multi-axis reduction until pytorch has it natively
    x = x.transpose(0,1).contiguous().view(x.size(1),-1)
    return x.mean(1), x.std(1)

noop = lambda x: x

def xy_transform(x_tfm=None, y_tfm=None):
    if x_tfm is None: x_tfm = noop
    if y_tfm is None: y_tfm = noop
    return lambda b: (x_tfm(b[0]), y_tfm(b[1]))

def xy_transforms(x_tfms=None, y_tfms=None):
    x_tfms = listify(x_tfms)
    if y_tfms is None: y_tfms=noop
    y_tfms = listify(y_tfms, x_tfms)
    return list(map(xy_transform, x_tfms, y_tfms))

def normalize(mean,std,x): return (x-mean.reshape(3,1,1))/std.reshape(3,1,1)
def denormalize(mean,std,x): return x * std.reshape(3,1,1) + mean.reshape(3,1,1)

TfmType = IntEnum('TfmType', 'Lighting Coord Affine Pixel Final')

def logit(x): return (x/(1-x)).log()
def logit_(x): return (x.div_(1-x)).log_()

def apply_lighting_tfm(func): return lambda x: func(logit_(x)).sigmoid()

def uniform(low=0, high=1, size=None):
    return random.uniform(low,high) if size is None else torch.FloatTensor(*size).uniform_(low,high)

def log_uniform(low, high, size=None):
    res = uniform(log(low), log(high), size)
    return exp(res) if size is None else res.exp_()

def randint(low, high, size=None):
    return random.randint(low,high) if size is None else torch.LongTensor(*size).random_(low,high+1)

def rand_bool(p=1, size=None): return uniform(0,1,size)<p

def resolve_args(func, **kwargs):
    for k,v in func.__annotations__.items():
        arg = listify(kwargs.get(k))
        if k != 'return': kwargs[k] = v(*arg)
    return kwargs

def make_p_func(func):
    return lambda x, *args, p, **kwargs: func(x,*args,**kwargs) if p else x

def make_tfm_func(func):
    def _inner(**kwargs):
        res = lambda: partial(make_p_func(func), **resolve_args(func, **kwargs))
        res.__annotations__ = func.__annotations__
        res.__annotations__['p'] = rand_bool
        return res
    return _inner

def reg_transform(func):
    setattr(sys.modules[func.__module__], f'{func.__name__}_tfm', make_tfm_func(func))
    return func

def resolve_tfms(tfms): return [f() for f in listify(tfms)]
@reg_transform
def brightness(x, change: uniform) -> TfmType.Lighting:  return x.add_(scipy.special.logit(change))

@reg_transform
def contrast(x, scale: log_uniform) -> TfmType.Lighting: return x.mul_(scale)

def grid_sample_nearest(input, coords, padding_mode='zeros'):
    if padding_mode=='border': coords.clamp(-1,1)
    bs,ch,h,w = input.size()
    sz = torch.tensor([w,h]).float()[None,None]
    coords.add_(1).mul_(sz/2)
    coords = coords[0].round_().long()
    if padding_mode=='zeros':
        mask = (coords[...,0] < 0) + (coords[...,1] < 0) + (coords[...,0] >= w) + (coords[...,1] >= h)
        mask.clamp_(0,1)
    coords[...,0].clamp_(0,w-1)
    coords[...,1].clamp_(0,h-1)
    result = input[...,coords[...,1],coords[...,0]]
    if padding_mode=='zeros': result[...,mask] = result[...,mask].zero_()
    return result

def grid_sample(x, coords, mode='bilinear', padding_mode='reflect'):
    if mode=='nearest': return grid_sample_nearest(x[None], coords, padding_mode)[0]
    if padding_mode=='reflect': # Reflect padding isn't implemented in grid_sample yet
        coords[coords < -1] = coords[coords < -1].mul_(-1).add_(-2)
        coords[coords > 1] = coords[coords > 1].mul_(-1).add_(2)
        padding_mode='zeros'
    return F.grid_sample(x[None], coords, mode=mode, padding_mode=padding_mode)[0]

def affine_grid(x, matrix, size=None):
    return F.affine_grid(matrix[None,:2], torch.Size((1,)+size))

def eye_new(x, n): return torch.eye(n, out=x.new_empty((n,n)))

def do_affine(img, m=None, func=None, size=None, **kwargs):
    if size is None: size = img.size()
    elif isinstance(size, int): size=(img.size(0),size,size)
    if m is None:
        if img.shape==size: return img
        else: m=eye_new(img, 3)
    c = affine_grid(img,  img.new_tensor(m), size=size)
    if func is not None: c = func(c)
    return grid_sample(img, c, **kwargs)

def affines_mat(matrices=None):
    if matrices is None or len(matrices)==0: return None
    matrices = [FloatTensor(m) for m in matrices if m is not None]
    return reduce(torch.matmul, matrices, torch.eye(3))

def make_p_affine(func):
    return lambda *args, p, **kwargs: func(*args,**kwargs) if p else None

def make_tfm_affine(func):
    def _inner(**kwargs):
        res = lambda: make_p_affine(func)(**resolve_args(func, **kwargs))
        res.__annotations__ = func.__annotations__
        res.__annotations__['p'] = rand_bool
        return res
    return _inner

def apply_affine_tfm(matrices=None, func=None, **kwargs):
    return partial(do_affine, m=affines_mat(matrices), func=func, **kwargs)

def reg_affine(func):
    setattr(sys.modules[func.__module__], f'{func.__name__}_tfm', make_tfm_affine(func))
    return func

@reg_affine
def rotate(degrees: uniform) -> TfmType.Affine:
    angle = degrees * math.pi / 180
    return [[cos(angle), -sin(angle), 0.],
            [sin(angle),  cos(angle), 0.],
            [0.        ,  0.        , 1.]]

@reg_affine
def zoom(scale: uniform) -> TfmType.Affine:
    return [[1/scale, 0,       0.],
            [0,       1/scale, 0.],
            [0,       0,       1.]]

@reg_transform
def jitter(x, magnitude: uniform) -> TfmType.Coord:
    return x.add_((torch.rand_like(x)-0.5)*magnitude*2)

@reg_transform
def flip_lr(x) -> TfmType.Pixel: return x.flip(2)
