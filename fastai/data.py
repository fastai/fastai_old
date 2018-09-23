from .torch_core import *

__all__ = ['DataBunch', 'DeviceDataLoader']

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
               tfms:Collection[Callable]=tfms, num_workers:int=default_cpus, collate_fn:Callable=data_collate, **kwargs:Any):
        "Create DeviceDataLoader from `dataset` with `batch_size` and `shuffle`: processs using `num_workers`"
        return cls(DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, **kwargs),
                   device=device, tfms=tfms, collate_fn=collate_fn)

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
    def create(cls, train_ds, valid_ds, test_ds=None, path='.', bs=64, num_workers=default_cpus,
               tfms=None, device=None, size=None, **kwargs)->'DataBunch':
        "`DataBunch` factory. `bs` batch size, `ds_tfms` for `Dataset`, `tfms` for `DataLoader`"
        datasets = [train_ds,valid_ds]
        if test_ds is not None: datasets.append(test_ds)
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
