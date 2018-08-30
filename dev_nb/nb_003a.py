
        #################################################
        ### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
        #################################################

import nb_002
from nb_002c import *

import operator
from random import sample
from torch.utils.data.sampler import Sampler

class FilesDataset(Dataset):
    def __init__(self, fns, labels, classes=None):
        if classes is None: classes = list(set(labels))
        self.classes = classes
        self.class2idx = {v:k for k,v in enumerate(classes)}
        self.fns = np.array(fns)
        self.y = [self.class2idx[o] for o in labels]

    def __len__(self): return len(self.fns)

    def __getitem__(self,i): return open_image(self.fns[i]),self.y[i]

    @classmethod
    def from_folder(cls, folder, classes=None, test_pct=0.):
        if classes is None: classes = [cls.name for cls in find_classes(folder)]

        fns,labels = [],[]
        for cl in classes:
            fnames = get_image_files(folder/cl)
            fns += fnames
            labels += [cl] * len(fnames)

        if test_pct==0.: return cls(fns, labels, classes=classes)

        fns,labels = np.array(fns),np.array(labels)
        is_test = np.random.uniform(size=(len(fns),)) < test_pct
        return (cls(fns[~is_test], labels[~is_test], classes=classes),
                cls(fns[is_test], labels[is_test], classes=classes))

def affine_mult(c,m):
    if m is None: return c
    size = c.size()
    _,h,w,_ = size
    m[0,1] *= h/w
    m[1,0] *= w/h
    c = c.view(-1,2)
    c = torch.addmm(m[:2,2], c,  m[:2,:2].t())
    return c.view(size)

nb_002.affine_mult = affine_mult

class TfmCrop(TfmPixel): order=99

@TfmCrop
def crop_pad(x, size, padding_mode='reflect',
             row_pct:uniform = 0.5, col_pct:uniform = 0.5):
    size = listify(size,2)
    rows,cols = size
    if x.size(1)<rows or x.size(2)<cols:
        row_pad = max((rows-x.size(1)+1)//2, 0)
        col_pad = max((cols-x.size(2)+1)//2, 0)
        x = F.pad(x[None], (col_pad,col_pad,row_pad,row_pad), mode=padding_mode)[0]
    row = int((x.size(1)-rows+1)*row_pct)
    col = int((x.size(2)-cols+1)*col_pct)

    x = x[:, row:row+rows, col:col+cols]
    return x.contiguous() # without this, get NaN later - don't know why

def round_multiple(x, mult): return (int(x/mult+0.5)*mult)

def get_crop_target(target_px, target_aspect=None, mult=32):
    target_px = listify(target_px, 2)
    target_r,target_c = target_px
    if target_aspect:
        target_r = math.sqrt(target_r*target_c/target_aspect)
        target_c = target_r*target_aspect
    return round_multiple(target_r,mult),round_multiple(target_c,mult)

@partial(Transform, order=99)
def crop_pad(img, size=None, mult=32, padding_mode=None,
             row_pct:uniform = 0.5, col_pct:uniform = 0.5):
    aspect = img.aspect if hasattr(img, 'aspect') else 1.
    if not size and hasattr(img, 'size'): size = img.size
    if not padding_mode:
        if hasattr(img, 'sample_kwargs') and ('padding_mode' in img.sample_kwargs):
            padding_mode = img.sample_kwargs['padding_mode']
        else: padding_mode='reflect'
    if padding_mode=='zeros': padding_mode='constant'

    rows,cols = get_crop_target(size, aspect, mult=mult)
    x = img.px
    if x.size(1)<rows or x.size(2)<cols:
        row_pad = max((rows-x.size(1)+1)//2, 0)
        col_pad = max((cols-x.size(2)+1)//2, 0)
        x = F.pad(x[None], (col_pad,col_pad,row_pad,row_pad), mode=padding_mode)[0]
    row = int((x.size(1)-rows+1)*row_pct)
    col = int((x.size(2)-cols+1)*col_pct)

    x = x[:, row:row+rows, col:col+cols]
    img.px = x.contiguous() # without this, get NaN later - don't know why
    return img

def get_resize_target(img, crop_target, do_crop=False):
    if crop_target is None: return None
    ch,r,c = img.shape
    target_r,target_c = crop_target
    ratio = (min if do_crop else max)(r/target_r, c/target_c)
    return ch,round(r/ratio),round(c/ratio)

@partial(Transform, order=TfmAffine.order-2)
def resize_image(x, *args, **kwargs): return x.resize(*args, **kwargs)

def _resize(self, size=None, do_crop=False, mult=32):
    assert self._flow is None
    if not size and hasattr(self, 'size'): size = self.size
    aspect = self.aspect if hasattr(self, 'aspect') else None
    crop_target = get_crop_target(size, aspect, mult=mult)
    target = get_resize_target(self, crop_target, do_crop)
    self.flow = affine_grid(target)
    return self

Image.resize=_resize

def is_listy(x)->bool: return isinstance(x, (tuple,list))

def apply_tfms(tfms, x, do_resolve=True, xtra=None, aspect=None, size=None,
               padding_mode='reflect', **kwargs):
    if not tfms: return x
    if not xtra: xtra={}
    tfms = sorted(listify(tfms), key=lambda o: o.tfm.order)
    if do_resolve: resolve_tfms(tfms)
    x = Image(x.clone())
    x.set_sample(padding_mode=padding_mode, **kwargs)
    x.aspect = aspect
    x.size = size

    for tfm in tfms:
        if tfm.tfm in xtra: x = tfm(x, **xtra[tfm.tfm])
        x = tfm(x)
    return x.px

def rand_zoom(*args, **kwargs): return zoom(*args, row_pct=(0,1), col_pct=(0,1), **kwargs)