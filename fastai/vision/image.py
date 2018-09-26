from ..torch_core import *
from ..data import *
import functools

_all__ = ['Image', 'ImageBBox', 'ImageBase', 'ImageMask', 'RandTransform', 'TfmAffine', 'TfmCoord', 'TfmCrop', 'TfmLighting', 
           'TfmPixel', 'Transform', 'affine_grid', 'affine_mult', 'apply_tfms', 'get_crop_target', 'get_default_args', 
           'get_resize_target', 'grid_sample', 'log_uniform', 'logit', 'logit_', 'rand_bool', 'rand_crop', 
           'rand_int', 'resolve_tfms', 'round_multiple', 'uniform', 'uniform_int']

def logit(x:Tensor)->Tensor:  return -(1/x-1).log()
def logit_(x:Tensor)->Tensor: return (x.reciprocal_().sub_(1)).log_().neg_()

def uniform(low:Number, high:Number=None, size:List[int]=None)->FloatOrTensor:
    "Draw 1 or shape=`size` random floats from uniform dist: min=`low`, max=`high`"
    if high is None: high=low
    return random.uniform(low,high) if size is None else torch.FloatTensor(*listify(size)).uniform_(low,high)

def log_uniform(low, high, size=None)->FloatOrTensor:
    "Draw 1 or shape=`size` random floats from uniform dist: min=log(`low`), max=log(`high`)"
    res = uniform(log(low), log(high), size)
    return exp(res) if size is None else res.exp_()

def rand_bool(p:float, size=None)->BoolOrTensor:
    "Draw 1 or shape=`size` random booleans (True occuring probability p)"
    return uniform(0,1,size)<p

def rand_int(low:int,high:int)->int: return random.randint(low, high)

def uniform_int(low:Number, high:Number, size:Optional[List[int]]=None)->FloatOrTensor:
    "Generate int or tensor `size` of ints from uniform(`low`,`high`)"
    return random.randint(low,high) if size is None else torch.randint(low,high,size)

def get_default_args(func:Callable):
    return {k: v.default
            for k, v in inspect.signature(func).parameters.items()
            if v.default is not inspect.Parameter.empty}

class ImageBase(ItemBase):
    "Img based `Dataset` items derive from this. Subclass to handle lighting, pixel, etc"
    def lighting(self, func:LightingFunc, *args, **kwargs)->'ImageBase': return self
    def pixel(self, func:PixelFunc, *args, **kwargs)->'ImageBase': return self
    def coord(self, func:CoordFunc, *args, **kwargs)->'ImageBase': return self
    def affine(self, func:AffineFunc, *args, **kwargs)->'ImageBase': return self

    def set_sample(self, **kwargs)->'ImageBase':
        "Set parameters that control how we `grid_sample` the image after transforms are applied"
        self.sample_kwargs = kwargs
        return self

    def clone(self)->'ImageBase':
        "Clones this item and its `data`"
        return self.__class__(self.data.clone())

class Image(ImageBase):
    "Supports appying transforms to image data"
    def __init__(self, px)->'Image':
        "create from raw tensor image data `px`"
        self._px = px
        self._logit_px=None
        self._flow=None
        self._affine_mat=None
        self.sample_kwargs = {}

    @property
    def shape(self)->Tuple[int,int,int]:
        "Returns (ch, h, w) for this image"
        return self._px.shape
    @property
    def size(self)->Tuple[int,int]:
        "Returns (h, w) for this image"
        return self.shape[-2:]
    @property
    def device(self)->torch.device: return self._px.device

    def __repr__(self): return f'{self.__class__.__name__} ({self.shape})'
    def _repr_jpeg_(self): return self._image2bytestr()

    def refresh(self)->None:
        "Applies any logit or affine transfers that have been "
        if self._logit_px is not None:
            self._px = self._logit_px.sigmoid_()
            self._logit_px = None
        if self._affine_mat is not None or self._flow is not None:
            self._px = grid_sample(self._px, self.flow, **self.sample_kwargs)
            self.sample_kwargs = {}
            self._flow = None
        return self

    @property
    def px(self)->TensorImage:
        "Get the tensor pixel buffer"
        self.refresh()
        return self._px
    @px.setter
    def px(self,v:TensorImage)->None:
        "Set the pixel buffer to `v`"
        self._px=v

    @property
    def flow(self)->FlowField:
        "Access the flow-field grid after applying queued affine transforms"
        if self._flow is None:
            self._flow = affine_grid(self.shape)
        if self._affine_mat is not None:
            self._flow = affine_mult(self._flow,self._affine_mat)
            self._affine_mat = None
        return self._flow

    @flow.setter
    def flow(self,v:FlowField): self._flow=v

    def lighting(self, func:LightingFunc, *args:Any, **kwargs:Any)->'Image':
        "Equivalent to `image = sigmoid(func(logit(image)))`"
        self.logit_px = func(self.logit_px, *args, **kwargs)
        return self

    def pixel(self, func:PixelFunc, *args, **kwargs)->'Image':
        "Equivalent to `image.px = func(image.px)`"
        self.px = func(self.px, *args, **kwargs)
        return self

    def coord(self, func:CoordFunc, *args, **kwargs)->'Image':
        "Equivalent to `image.flow = func(image.flow, image.size)`"
        self.flow = func(self.flow, self.shape, *args, **kwargs)
        return self

    def affine(self, func:AffineFunc, *args, **kwargs)->'Image':
        "Equivalent to `image.affine_mat = image.affine_mat @ func()`"
        m = tensor(func(*args, **kwargs)).to(self.device)
        self.affine_mat = self.affine_mat @ m
        return self

    def resize(self, size:Union[int,TensorImageSize])->'Image':
        "Resize the image to `size`, size can be a single int"
        assert self._flow is None
        if isinstance(size, int): size=(self.shape[0], size, size)
        self.flow = affine_grid(size)
        return self

    @property
    def affine_mat(self)->AffineMatrix:
        "Get the affine matrix that will be applied by `refresh`"
        if self._affine_mat is None:
            self._affine_mat = torch.eye(3).to(self.device)
        return self._affine_mat
    @affine_mat.setter
    def affine_mat(self,v)->None: self._affine_mat=v

    @property
    def logit_px(self)->LogitTensorImage:
        "Get logit(image.px)"
        if self._logit_px is None: self._logit_px = logit_(self.px)
        return self._logit_px
    @logit_px.setter
    def logit_px(self,v:LogitTensorImage)->None: self._logit_px=v

    def show(self, ax:plt.Axes=None, **kwargs:Any)->None:
        "Plots the image into `ax`"
        show_image(self.px, ax=ax, **kwargs)

    @property
    def data(self)->TensorImage:
        "Returns this images pixels as a tensor"
        return self.px

class ImageMask(Image):
    "Class for image segmentation target"
    def lighting(self, func:LightingFunc, *args:Any, **kwargs:Any)->'Image': return self

    def refresh(self):
        self.sample_kwargs['mode'] = 'nearest'
        return super().refresh()

class ImageBBox(ImageMask):
    "Image class for bbox-style annotations"
    def clone(self):
        return self.__class__(self.px.clone())

    @classmethod
    def create(cls, bboxes:Collection[Collection[int]], h:int, w:int) -> 'ImageBBox':
        "Creates an ImageBBox object from bboxes"
        pxls = torch.zeros(len(bboxes),h, w).long()
        for i,bbox in enumerate(bboxes):
            pxls[i,bbox[0]:bbox[2]+1,bbox[1]:bbox[3]+1] = 1
        return cls(pxls)

    @property
    def data(self) -> LongTensor:
        bboxes = []
        for i in range(self.px.size(0)):
            idxs = torch.nonzero(self.px[i])
            if len(idxs) != 0:
                bboxes.append(torch.tensor([idxs[:,0].min(), idxs[:,1].min(), idxs[:,0].max(), idxs[:,1].max()])[None])
        return torch.cat(bboxes, 0).squeeze()

class Transform():
    "Utility class for adding probability and wrapping support to transform funcs"
    _wrap=None
    order=0
    def __init__(self, func:Callable, order:Optional[int]=None)->None:
        "Create a transform for `func` and assign it an priority `order`, attach to Image class"
        if order is not None: self.order=order
        self.func=func
        functools.update_wrapper(self, self.func)
        self.func.__annotations__['return'] = Image
        self.params = copy(func.__annotations__)
        self.def_args = get_default_args(func)
        setattr(Image, func.__name__,
                lambda x, *args, **kwargs: self.calc(x, *args, **kwargs))

    def __call__(self, *args:Any, p:float=1., is_random:bool=True, **kwargs:Any)->Image:
        "Calc now if `args` passed; else create a transform called prob `p` if `random`"
        if args: return self.calc(*args, **kwargs)
        else: return RandTransform(self, kwargs=kwargs, is_random=is_random, p=p)

    def calc(self, x:Image, *args:Any, **kwargs:Any)->Image:
        "Apply to image `x`, wrapping it if necessary"
        if self._wrap: return getattr(x, self._wrap)(self.func, *args, **kwargs)
        else:          return self.func(x, *args, **kwargs)

    @property
    def name(self)->str: return self.__class__.__name__

    def __repr__(self)->str: return f'{self.name} ({self.func.__name__})'

TfmList = Union[Transform, Collection[Transform]]
Tfms = Optional[TfmList]

@dataclass
class RandTransform():
    "Wraps `Transform` to add randomized execution"
    tfm:Transform
    kwargs:dict
    p:int=1.0
    resolved:dict = field(default_factory=dict)
    do_run:bool = True
    is_random:bool = True
    def __post_init__(self): functools.update_wrapper(self, self.tfm)

    def resolve(self)->None:
        "Bind any random variables needed tfm calc"
        if not self.is_random:
            self.resolved = {**self.tfm.def_args, **self.kwargs}
            return

        self.resolved = {}
        # for each param passed to tfm...
        for k,v in self.kwargs.items():
            # ...if it's annotated, call that fn...
            if k in self.tfm.params:
                rand_func = self.tfm.params[k]
                self.resolved[k] = rand_func(*listify(v))
            # ...otherwise use the value directly
            else: self.resolved[k] = v
        # use defaults for any args not filled in yet
        for k,v in self.tfm.def_args.items():
            if k not in self.resolved: self.resolved[k]=v
        # anything left over must be callable without params
        for k,v in self.tfm.params.items():
            if k not in self.resolved and k!='return': self.resolved[k]=v()

        self.do_run = rand_bool(self.p)

    @property
    def order(self)->int: return self.tfm.order

    def __call__(self, x:Image, *args, **kwargs)->Image:
        "Randomly execute our tfm on `x`"
        return self.tfm(x, *args, **{**self.resolved, **kwargs}) if self.do_run else x


def resolve_tfms(tfms:TfmList):
    "Resolve every tfm in `tfms`"
    for f in listify(tfms): f.resolve()

def grid_sample(x:TensorImage, coords:FlowField, mode:str='bilinear', padding_mode:str='reflect')->TensorImage:
    "Grab pixels in `coords` from `input` sampling by `mode`. pad is reflect or zeros."
    if padding_mode=='reflect': padding_mode='reflection'
    return F.grid_sample(x[None], coords, mode=mode, padding_mode=padding_mode)[0]

def affine_grid(size:TensorImageSize)->FlowField:
    size = ((1,)+size)
    N, C, H, W = size
    grid = FloatTensor(N, H, W, 2)
    linear_points = torch.linspace(-1, 1, W) if W > 1 else tensor([-1])
    grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(grid[:, :, :, 0])
    linear_points = torch.linspace(-1, 1, H) if H > 1 else tensor([-1])
    grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(grid[:, :, :, 1])
    return grid

def affine_mult(c:FlowField,m:AffineMatrix)->FlowField:
    "Multiply `c` by `m` - can adjust for rectangular shaped `c`"
    if m is None: return c
    size = c.size()
    _,h,w,_ = size
    m[0,1] *= h/w
    m[1,0] *= w/h
    c = c.view(-1,2)
    c = torch.addmm(m[:2,2], c,  m[:2,:2].t())
    return c.view(size)

class TfmAffine(Transform):
    "Decorator for affine tfm funcs"
    order,_wrap = 5,'affine'
class TfmPixel(Transform):
    "Decorator for pixel tfm funcs"
    order,_wrap = 10,'pixel'
class TfmCoord(Transform): 
    "Decorator for coord tfm funcs"
    order,_wrap = 4,'coord'
class TfmCrop(TfmPixel):
    "Decorator for crop tfm funcs"
    order=99
class TfmLighting(Transform): 
    "Decorator for lighting tfm funcs"
    order,_wrap = 8,'lighting'

def round_multiple(x:int, mult:int)->int:
    "Calc `x` to nearest multiple of `mult`"
    return (int(x/mult+0.5)*mult)

def get_crop_target(target_px:Union[int,Tuple[int,int]], mult:int=32)->Tuple[int,int]:
    "Calc crop shape of `target_px` to nearest multiple of `mult`"
    target_r,target_c = listify(target_px, 2)
    return round_multiple(target_r,mult),round_multiple(target_c,mult)

def get_resize_target(img, crop_target, do_crop=False)->TensorImageSize:
    "Calc size of `img` to fit in `crop_target` - adjust based on `do_crop`"
    if crop_target is None: return None
    ch,r,c = img.shape
    target_r,target_c = crop_target
    ratio = (min if do_crop else max)(r/target_r, c/target_c)
    return ch,round(r/ratio),round(c/ratio)

def apply_tfms(tfms:TfmList, x:TensorImage, do_resolve:bool=True,
               xtra:Optional[Dict[Transform,dict]]=None, size:Optional[Union[int,TensorImageSize]]=None,
               mult:int=32, do_crop:bool=True, padding_mode:str='reflect', **kwargs:Any)->TensorImage:
    "Apply all `tfms` to `x` - `do_resolve`: bind random args - size,mult used to crop/pad"
    if tfms or xtra or size:
        if not xtra: xtra={}
        tfms = sorted(listify(tfms), key=lambda o: o.tfm.order)
        if do_resolve: resolve_tfms(tfms)
        x = x.clone()
        x.set_sample(padding_mode=padding_mode, **kwargs)
        if size:
            crop_target = get_crop_target(size, mult=mult)
            target = get_resize_target(x, crop_target, do_crop=do_crop)
            x.resize(target)

        size_tfms = [o for o in tfms if isinstance(o.tfm,TfmCrop)]
        for tfm in tfms:
            if tfm.tfm in xtra: x = tfm(x, **xtra[tfm.tfm])
            elif tfm in size_tfms: x = tfm(x, size=size, padding_mode=padding_mode)
            else: x = tfm(x)
    return x
