from .imports.core import *
from .imports.vision import *
from .imports.torch import *
from . import core as c
from . import torch_core as tc

def find_classes(folder:Path) -> Collection[Path]:
    "Find all the classes in a given folder"
    classes = [d for d in folder.iterdir()
               if d.is_dir() and not d.name.startswith('.')]
    assert(len(classes)>0)
    return sorted(classes, key=lambda d: d.name)

def get_image_files(c:Path) -> Collection[Path]:
    "Find all the image files in a given foldder"
    return [o for o in list(c.iterdir())
            if not o.name.startswith('.') and not o.is_dir()]

def pil2tensor(image:Image) -> Tensor:
    "Transforms a PIL Image in a torch tensor"
    arr = ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    arr = arr.view(image.size[1], image.size[0], -1)
    arr = arr.permute(2,0,1)
    return arr.float().div_(255)

class FilesDataset(Dataset):
    "Basic class to create a dataset from folders and files"

    def __init__(self, fns:Collection[c.FileLike], labels:Collection[str], classes:Collection[str]=None):
        if classes is None: classes = list(set(labels))
        self.classes = classes
        self.class2idx = {v:k for k,v in enumerate(classes)}
        self.fns = np.array(fns)
        self.y = [self.class2idx[o] for o in labels]
        
    def __len__(self) -> int: return len(self.fns)

    def __getitem__(self,i:int) -> Tuple[Tensor, Any]:
        x = Image.open(self.fns[i]).convert('RGB')
        return pil2tensor(x),self.y[i]
    
    def __repr__(self) -> str:
        return f'FilesDataset, length {len(self.fns)}, {len(self.classes)} classes: {str(self.classes)}'

    @classmethod
    def from_folder(cls, folder:Path, classes:Collection[str]=None, test_pct:float=0.):
        if classes is None: classes = [cls.name for cls in find_classes(folder)]
            
        fns,labels = [],[]
        for cl in classes:
            fnames = get_image_files(folder/cl)
            fns += fnames
            labels += [cl] * len(fnames)
            
        if test_pct==0.: return cls(fns, labels)
        
        fns,labels = np.array(fns),np.array(labels)
        is_test = np.random.uniform(size=(len(fns),)) < test_pct
        return cls(fns[~is_test], labels[~is_test]), cls(fns[is_test], labels[is_test])

@dataclass
class DeviceDataLoader():
    "Wrapper around the dataloader that puts the tensors on the GPU and handles FP16 precision"

    dl:DataLoader
    device:torch.device
    half:bool = False
        
    def __len__(self) -> int: return len(self.dl)
    def __iter__(self) -> Iterator:
        self.gen = (tc.to_device(self.device,o) for o in self.dl)
        if self.half: self.gen = (tc.to_half(o) for o in self.gen)
        return iter(self.gen)

    def __repr__(self) -> str:
        return f'DeviceDataLoader, batch size={self.dl.batch_size}, {len(self.dl)} batches on {self.device}, FP16: {self.half}'

    @classmethod
    def create(cls, *args, device:torch.device=tc.default_device, **kwargs):
        return cls(DataLoader(*args, **kwargs), device=device, half=False)

class DataBunch():
    "Data object that regroups training and validation data"
    
    def __init__(self, train_ds:Dataset, valid_ds:Dataset, bs:int=64, device:torch.device=None, num_workers:int=4):
        self.device,self.bs = tc.default_device if device is None else device,bs
        self.train_dl = DeviceDataLoader.create(train_ds, bs, shuffle=True, num_workers=num_workers, device=self.device)
        self.valid_dl = DeviceDataLoader.create(valid_ds, bs*2, shuffle=False, num_workers=num_workers, device=self.device)

    def __repr__(self) -> str:
        res = f'DataBunch, batch_size={self.bs} on {self.device}.\n  train dataloader: {len(self.train_dl)} batches'
        if self.valid_dl is not None: res += f'\n  validation dataloader: {len(self.valid_dl)} batches'
        return res

    #TODO: uncomment when transforms are available
    #@classmethod
    #def create(cls, train_ds, valid_ds, train_tfm=None, valid_tfm=None, **kwargs):
    #    return cls(TfmDataset(train_ds, train_tfm), TfmDataset(valid_ds, valid_tfm))
        
    @property
    def train_ds(self) -> Dataset: return self.train_dl.dl.dataset
    @property
    def valid_ds(self) -> Dataset: return self.valid_dl.dl.dataset