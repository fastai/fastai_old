from .imports.core import *
from .imports.vision import *
from .imports.torch import *
from . import core as c

default_device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

def find_classes(folder):
    classes = [d for d in folder.iterdir()
               if d.is_dir() and not d.name.startswith('.')]
    assert(len(classes)>0)
    return sorted(classes, key=lambda d: d.name)

def get_image_files(c):
    return [o for o in list(c.iterdir())
            if not o.name.startswith('.') and not o.is_dir()]

def pil2tensor(image):
    arr = ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    arr = arr.view(image.size[1], image.size[0], -1)
    arr = arr.permute(2,0,1)
    return arr.float().div_(255)

class FilesDataset(Dataset):
    def __init__(self, fns, labels, classes=None):
        if classes is None: classes = list(set(labels))
        self.classes = classes
        self.class2idx = {v:k for k,v in enumerate(classes)}
        self.fns = np.array(fns)
        self.y = [self.class2idx[o] for o in labels]
        
    def __len__(self): return len(self.fns)

    def __getitem__(self,i):
        x = Image.open(self.fns[i]).convert('RGB')
        return pil2tensor(x),self.y[i]
    
    @classmethod
    def from_folder(cls, folder, classes=None, test_pct=0.):
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
    dl: DataLoader
    device: torch.device
    progress_func: Callable
    half: bool = False
        
    def __len__(self): return len(self.dl)
    def __iter__(self):
        self.gen = (c.to_device(self.device,o) for o in self.dl)
        if self.half: self.gen = (c.to_half(o) for o in self.gen)
        if self.progress_func is not None:
            self.gen = self.progress_func(self.gen, total=len(self.dl), leave=False)
        return iter(self.gen)

    @classmethod
    def create(cls, *args, device=default_device, progress_func=c.tqdm, **kwargs):
        return cls(DataLoader(*args, **kwargs), device=device, progress_func=progress_func, half=False)

class DataBunch():
    def __init__(self, train_ds, valid_ds, bs=64, device=None, num_workers=4):
        self.device = default_device if device is None else device
        self.train_dl = DeviceDataLoader.create(train_ds, bs, shuffle=True, num_workers=num_workers)
        self.valid_dl = DeviceDataLoader.create(valid_ds, bs*2, shuffle=False, num_workers=num_workers)

    #TODO: uncomment when transforms are available
    #@classmethod
    #def create(cls, train_ds, valid_ds, train_tfm=None, valid_tfm=None, **kwargs):
    #    return cls(TfmDataset(train_ds, train_tfm), TfmDataset(valid_ds, valid_tfm))
        
    @property
    def train_ds(self): return self.train_dl.dl.dataset
    @property
    def valid_ds(self): return self.valid_dl.dl.dataset