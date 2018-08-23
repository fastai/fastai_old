from .torch_core import *

__all__ = [DeviceDataLoader, DataBunch]

@dataclass
class DeviceDataLoader():
    "Wrapper around the dataloader that puts the tensors on the GPU and handles FP16 precision"
    dl:DataLoader
    device:torch.device
    half:bool = False

    def __len__(self) -> int: return len(self.dl)
    def __iter__(self) -> Iterator:
        self.gen = (to_device(self.device,o) for o in self.dl)
        if self.half: self.gen = (to_half(o) for o in self.gen)
        return iter(self.gen)

    def __repr__(self) -> str:
        return f'DeviceDataLoader, batch size={self.dl.batch_size}, {len(self.dl)} batches on {self.device}, FP16: {self.half}'

    @classmethod
    def create(cls, *args, device:torch.device=default_device, **kwargs):
        return cls(DataLoader(*args, **kwargs), device=device, half=False)

class DataBunch():
    "Data object that regroups training and validation data"

    def __init__(self, train_ds:Dataset, valid_ds:Dataset, bs:int=64, device:torch.device=None, num_workers:int=4):
        self.device,self.bs = default_device if device is None else device,bs
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

