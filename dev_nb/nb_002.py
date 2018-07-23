from nb_001b import *
from PIL import Image
import PIL, matplotlib.pyplot as plt
from torch.utils.data import Dataset
from operator import itemgetter, attrgetter
from numpy import random


def find_classes(folder):
    classes = [d for d in folder.iterdir()
               if d.is_dir() and not d.name.startswith('.')]
    assert(len(classes)>0)
    return sorted(classes, key=lambda d: d.name)

def get_image_files(c):
    return [o for o in list(c.iterdir())
            if not o.name.startswith('.') and not o.is_dir()]

def pil2tensor(image):
    arr = np.array(image, dtype=np.float32)/255.
    if len(arr.shape)==2: arr = np.repeat(arr[...,None],3,2)
    return torch.from_numpy(arr).permute(2,0,1).contiguous()


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


def log_uniform(low, high):
    return np.exp(random.uniform(np.log(low), np.log(high)))


def func_rand_kw(func, kwargs):
    return {k:func.__annotations__[k](*v) for k,v in kwargs.items()}

def func_rand(func, **kwargs):
    return lambda x: func(x, **func_rand_kw(func, kwargs))


def reg_transform(tfm_type):
    def decorator(func):
        func.tfm_type = tfm_type
        return func
    return decorator


@reg_transform('pixel')
def brightness(x, scale: log_uniform): return x.mul_(scale).clamp_(0,1)


def grid_sample(x, coords, padding='reflect'):
    if padding=='reflect': # Reflect padding isn't implemented in grid_sample yet
        coords[coords < -1] = coords[coords < -1].mul_(-1).add_(-2)
        coords[coords > 1] = coords[coords > 1].mul_(-1).add_(2)
        padding='zeros'
    return F.grid_sample(x[None], coords, padding_mode=padding)[0]


def affine_coords(x, matrix):
    return F.affine_grid(matrix[None,:2], x[None].size())

def do_affine(img, m):
    c = affine_coords(img,  img.new_tensor(m))
    return grid_sample(img, c, padding='zeros')


def rotate_matrix(degrees):
    angle = degrees * math.pi / 180
    return [[math.cos(angle), -math.sin(angle), 0.],
            [math.sin(angle),  math.cos(angle), 0.],
            [0.             ,  0.             , 1.]]

def zoom_matrix(zoom):
    return [[zoom, 0,    0.],
            [0,    zoom, 0.],
            [0,    0   , 1.]]


def eye_like(x, n): return torch.eye(n, out=x.new_empty((n,n)))

def affines_mat(x, matrices):
    matrices = list(map(x.new_tensor, matrices))
    return reduce(torch.matmul, matrices, eye_like(x, 3))


