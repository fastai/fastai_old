from nb_001b import *
import sys, PIL, matplotlib.pyplot as plt, itertools, math, collections, torch
import scipy.stats, scipy.special
from enum import Enum, IntEnum
from torch.utils.data import Dataset
from operator import itemgetter, attrgetter
from numpy import random, cos, sin, tan, tanh
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
    def __init__(self, folder, classes):
        self.fns, self.y = [], []
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

def show_image(img, ax=None, figsize=(3,3)):
    if ax is None: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(image2np(img))
    ax.axis('off')

def show_image_batch(dl, classes, rows=None):
    if rows is None: rows = int(math.sqrt(len(x)))
    x,y = next(iter(dl))[:rows*rows]
    show_images(x,y,rows, classes)

def show_images(x,y,rows, classes):
    fig, axs = plt.subplots(rows,rows,figsize=(12,15))
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

def denorm(x): return x * data_std.reshape(3,1,1) + data_mean.reshape(3,1,1)

TfmType = IntEnum('TfmType', 'Pixel Coord Affine')

def log_uniform(low, high): return np.exp(random.uniform(np.log(low), np.log(high)))

def logit(x): return (x/(1-x)).log()
def logit_(x): return (x.div_(1-x)).log_()

def apply_pixel_tfm(func): return lambda x: func(logit_(x)).sigmoid()

def resolve_args(func, **kwargs):
    return {k:v(*kwargs[k]) for k,v in func.__annotations__.items() if k != 'return'}

def copy_anno(func, new_func):
    def _inner(*args, **kwargs):
        res = new_func(*args, **kwargs)
        res.__annotations__ = func.__annotations__
        return res
    return _inner

def make_tfm_func(func):
    res = lambda **kwargs: lambda: partial(func, **resolve_args(func, **kwargs))
    return copy_anno(func, res)

def reg_transform(func):
    setattr(sys.modules[__name__], f'{func.__name__}_tfm', make_tfm_func(func))
    return func

def resolve_tfms(tfms): return [f() for f in listify(tfms)]
def compose_tfms(tfms): return compose(resolve_tfms(tfms))
def apply_pixel_tfms(tfms): return apply_pixel_tfm(compose_tfms(tfms))

@reg_transform
def brightness(x, change: random.uniform) -> TfmType.Pixel:
    return x.add_(scipy.special.logit(change))

@reg_transform
def contrast(x, scale: log_uniform) -> TfmType.Pixel:
    return x.mul_(scale)

def grid_sample(x, coords, padding='reflect'):
    if padding=='reflect': # Reflect padding isn't implemented in grid_sample yet
        coords[coords < -1] = coords[coords < -1].mul_(-1).add_(-2)
        coords[coords > 1] = coords[coords > 1].mul_(-1).add_(2)
        padding='zeros'
    return F.grid_sample(x[None], coords, padding_mode=padding)[0]

def affine_grid(x, matrix): return F.affine_grid(matrix[None,:2], x[None].size())

def do_affine(img, m=None, func=None):
    if m is None: m=eye_new(img, 3)
    c = affine_grid(img,  img.new_tensor(m))
    if func is not None: c = func(c)
    return grid_sample(img, c, padding='zeros')

def eye_new(x, n): return torch.eye(n, out=x.new_empty((n,n)))

def affines_mat(matrices):
    matrices = list(map(torch.FloatTensor, matrices))
    return reduce(torch.matmul, matrices, torch.eye(3))

def make_tfm_affine(func):
    res = lambda **kwargs: lambda: func(**resolve_args(func, **kwargs))
    return copy_anno(func, res)

def compose_affine_tfms(affine_funcs=None, funcs=None):
    matrices = resolve_tfms(affine_funcs)
    return partial(do_affine, m=affines_mat(matrices), func=compose_tfms(funcs))

def reg_affine(func):
    setattr(sys.modules[__name__], f'{func.__name__}_tfm', make_tfm_affine(func))
    return func

@reg_affine
def rotate(degrees: random.uniform) -> TfmType.Affine:
    angle = degrees * math.pi / 180
    return [[cos(angle), -sin(angle), 0.],
            [sin(angle),  cos(angle), 0.],
            [0.        ,  0.        , 1.]]

@reg_affine
def zoom(scale: random.uniform) -> TfmType.Affine:
    return [[scale, 0,     0.],
            [0,     scale, 0.],
            [0,     0   ,  1.]]

@reg_transform
def jitter(x, magnitude: random.uniform) -> TfmType.Coord:
    return x.add_((torch.rand_like(x)-0.5)*magnitude*2)

def dict_groupby(iterable, key=None):
    return {k:list(v) for k,v in itertools.groupby(sorted(iterable, key=key), key=key)}

def resolve_pipeline(tfms):
    tfms = listify(tfms)
    if len(tfms)==0: return noop
    grouped_tfms = dict_groupby(tfms, lambda o: o.__annotations__['return'])
    pixel_tfms,coord_tfms,affine_tfms = map(grouped_tfms.get, TfmType)
    pixel_tfm = apply_pixel_tfms(pixel_tfms)
    affine_tfm = compose_affine_tfms(affine_tfms, funcs=coord_tfms)
    return compose([pixel_tfm,affine_tfm])

