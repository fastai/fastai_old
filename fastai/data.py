from .torch_core import *
from .tabular.data import *

TfmList = Collection[Callable]

@dataclass
class DeviceDataLoader():
    "Binds a `DataLoader` to a `torch.device`"
    dl: DataLoader
    device: torch.device
    tfms: List[Callable]=None
    collate_fn: Callable=data_collate
    def __post_init__(self):
        self.dl.collate_fn=self.collate_fn
        self.tfms = listify(self.tfms)

    def __len__(self)->int: return len(self.dl)
    def __getattr__(self,k:str)->Any: return getattr(self.dl, k)

    def add_tfm(self,tfm:Callable)->None:    self.tfms.append(tfm)
    def remove_tfm(self,tfm:Callable)->None: self.tfms.remove(tfm)

    def proc_batch(self,b:Tensor)->Tensor:
        "Proces batch `b` of `TensorImage`"
        b = to_device(b, self.device)
        for f in listify(self.tfms): b = f(b)
        return b

    def __iter__(self):
        "Process and returns items from `DataLoader`"
        self.gen = map(self.proc_batch, self.dl)
        return iter(self.gen)

    @classmethod
    def create(cls, dataset:Dataset, bs:int=1, shuffle:bool=False, device:torch.device=default_device,
               tfms:TfmList=tfms, num_workers:int=default_cpus, collate_fn:Callable=data_collate, **kwargs:Any):
        "Create DeviceDataLoader from `dataset` with `batch_size` and `shuffle`: processs using `num_workers`"
        return cls(DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, **kwargs),
                   device=device, tfms=tfms, collate_fn=collate_fn)

class DatasetTfm(Dataset):
    "`Dataset` that applies a list of transforms to every item drawn"
    def __init__(self, ds:Dataset, tfms:TfmList=None, tfm_y:bool=False, **kwargs:Any):
        "this dataset will apply `tfms` to `ds`"
        self.ds,self.tfms,self.kwargs,self.tfm_y = ds,tfms,kwargs,tfm_y
        self.y_kwargs = {**self.kwargs, 'do_resolve':False}

    def __len__(self)->int: return len(self.ds)

    def __getitem__(self,idx:int)->Tuple[ItemBase,Any]:
        "returns tfms(x),y"
        x,y = self.ds[idx]
        x = apply_tfms(self.tfms, x, **self.kwargs)
        if self.tfm_y: y = apply_tfms(self.tfms, y, **self.y_kwargs)
        return x, y

    def __getattr__(self,k):
        "passthrough access to wrapped dataset attributes"
        return getattr(self.ds, k)

def transform_datasets(train_ds:Dataset, valid_ds:Dataset, test_ds:Optional[Dataset]=None,
                       tfms:Optional[Tuple[TfmList,TfmList]]=None, **kwargs:Any):
    "Create train, valid and maybe test DatasetTfm` using `tfms` = (train_tfms,valid_tfms)"
    res = [DatasetTfm(train_ds, tfms[0],  **kwargs),
           DatasetTfm(valid_ds, tfms[1],  **kwargs)]
    if test_ds is not None: res.append(DatasetTfm(test_ds, tfms[1],  **kwargs))
    return res

class DataBunch():
    "Bind `train_dl`,`valid_dl` and`test_dl` to `device`. tfms are DL tfms (normalize). `path` is for models."
    def __init__(self, train_dl:DataLoader, valid_dl:DataLoader, test_dl:Optional[DataLoader]=None,
                 device:torch.device=None, tfms:Optional[Collection[Callable]]=None, path:PathOrStr='.'):
        "Bind `train_dl`,`valid_dl` and`test_dl` to `device`. tfms are DL tfms (normalize). `path` is for models."
        self.device = default_device if device is None else device
        self.train_dl = DeviceDataLoader(train_dl, self.device, tfms=tfms)
        self.valid_dl = DeviceDataLoader(valid_dl, self.device, tfms=tfms)
        self.test_dl  = DeviceDataLoader(test_dl,  self.device, tfms=tfms) if test_dl else None
        self.path = Path(path)

    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None,
               path='.', bs=64, ds_tfms=None, num_workers=default_cpus,
               tfms=None, device=None, size=None, **kwargs)->'DataBunch':
        "`DataBunch` factory. `bs` batch size, `ds_tfms` for `Dataset`, `tfms` for `DataLoader`"
        datasets = [train_ds,valid_ds]
        if test_ds is not None: datasets.append(test_ds)
        if ds_tfms: datasets = transform_datasets(*datasets, tfms=ds_tfms, size=size, **kwargs)
        dls = [DataLoader(*o, num_workers=num_workers) for o in
               zip(datasets, (bs,bs*2,bs*2), (True,False,False))]
        return cls(*dls, path=path, device=device, tfms=tfms)

    def __getattr__(self,k)->Any: return getattr(self.train_ds, k)
    def holdout(self, is_test:bool=False)->DeviceDataLoader:
        "Returns correct holdout `Dataset` for test vs validation (`is_test`)"
        return self.test_dl if is_test else self.valid_dl

    @property
    def train_ds(self)->Dataset: return self.train_dl.dl.dataset
    @property
    def valid_ds(self)->Dataset: return self.valid_dl.dl.dataset
