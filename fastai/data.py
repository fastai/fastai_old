from .torch_core import *
from .vision.data import *
from .text.data import *
from .tabular.data import *

def normalize(x:TensorImage, mean:float,std:float)->TensorImage:   return (x-mean[...,None,None]) / std[...,None,None]
def denormalize(x:TensorImage, mean:float,std:float)->TensorImage: return x*std[...,None,None] + mean[...,None,None]

def normalize_batch(b:Tuple[Tensor,Tensor], mean:float, std:float, do_y:bool=False)->Tuple[Tensor,Tensor]:
    "`b` = `x`,`y` - normalize `x` array of imgs and `do_y` optionally `y`"
    x,y = b
    x = normalize(x,mean,std)
    if do_y: y = normalize(y,mean,std)
    return x,y

def normalize_funcs(mean:float, std, do_y=False, device=None)->[Callable,Callable]:
    "Create normalize/denormalize func using `mean` and `std`, can specify `do_y` and `device`"
    if device is None: device=default_device
    return (partial(normalize_batch, mean=mean.to(device),std=std.to(device)),
            partial(denormalize,     mean=mean,           std=std))

cifar_stats = (tensor([0.491, 0.482, 0.447]), tensor([0.247, 0.243, 0.261]))
cifar_norm,cifar_denorm = normalize_funcs(*cifar_stats)
imagenet_stats = tensor([0.485, 0.456, 0.406]), tensor([0.229, 0.224, 0.225])
imagenet_norm,imagenet_denorm = normalize_funcs(*imagenet_stats)

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

def data_from_imagefolder(path:PathOrStr, train:PathOrStr='train', valid:PathOrStr='valid',
                          test:Optional[PathOrStr]=None, **kwargs:Any):
    "Create `DataBunch` from imagenet style dataset in `path` with `train`,`valid`,`test` subfolders"
    path=Path(path)
    train_ds = FilesDataset.from_folder(path/train)
    datasets = [train_ds, FilesDataset.from_folder(path/valid, classes=train_ds.classes)]
    if test: datasets.append(FilesDataset.from_single_folder(
        path/test,classes=train_ds.classes))
    return DataBunch.create(*datasets, path=path, **kwargs)

DataFunc = Callable[[Collection[DatasetBase], PathOrStr, KWArgs], DataBunch]

def standard_data(datasets:Collection[DatasetBase], path:PathOrStr, **kwargs) -> DataBunch:
    "Simply creates a `DataBunch` from the `datasets`"
    return DataBunch.create(*datasets, path=path, **kwargs)

def lm_data(datasets:Collection[TextDataset], path:PathOrStr, **kwargs) -> DataBunch:
    "Creates a `DataBunch` from the `datasets` for language modelling"
    dataloaders = [LanguageModelLoader(ds, **kwargs) for ds in datasets]
    return DataBunch(*dataloaders, path=path)

def classifier_data(datasets:Collection[TextDataset], path:PathOrStr, **kwargs) -> DataBunch:
    "Function that transform the `datasets` in a `DataBunch` for classification"
    bs = kwargs.pop('bs') if 'bs' in kwargs else 64
    pad_idx = kwargs.pop('pad_idx') if 'pad_idx' in kwargs else 1
    train_sampler = SortishSampler(datasets[0].ids, key=lambda x: len(datasets[0].ids[x]), bs=bs//2)
    train_dl = DeviceDataLoader.create(datasets[0], bs//2, sampler=train_sampler, collate_fn=pad_collate)
    dataloaders = [train_dl]
    for ds in datasets[1:]:
        sampler = SortSampler(ds.ids, key=lambda x: len(ds.ids[x]))
        dataloaders.append(DeviceDataLoader.create(ds, bs,  sampler=sampler, collate_fn=pad_collate))
    return DataBunch(*dataloaders, path=path)

def data_from_textids(path:PathOrStr, train:str='train', valid:str='valid', test:Optional[str]=None,
                      data_func:DataFunc=standard_data, itos:str='itos.pkl', **kwargs) -> DataBunch:
    "Creates a `DataBunch` from ids, labels and a dictionary."
    path=Path(path)
    txt_kwargs, kwargs = extract_kwargs(['max_vocab', 'chunksize', 'min_freq', 'n_labels', 'id_suff', 'lbl_suff'], kwargs)
    train_ds = TextDataset.from_ids(path, train, itos=itos, **txt_kwargs)
    datasets = [train_ds, TextDataset.from_ids(path, valid, itos=itos, **txt_kwargs)]
    if test: datasets.append(TextDataset.from_ids(path, test, itos=itos, **txt_kwargs))
    return data_func(datasets, path, **kwargs)

def data_from_texttokens(path:PathOrStr, train:str='train', valid:str='valid', test:Optional[str]=None,
                         data_func:DataFunc=standard_data, vocab:Vocab=None, **kwargs) -> DataBunch:
    "Creates a `DataBunch` from tokens and labels."
    path=Path(path)
    txt_kwargs, kwargs = extract_kwargs(['max_vocab', 'chunksize', 'min_freq', 'n_labels', 'tok_suff', 'lbl_suff'], kwargs)
    train_ds = TextDataset.from_tokens(path, train, vocab=vocab, **txt_kwargs)
    datasets = [train_ds, TextDataset.from_tokens(path, valid, vocab=train_ds.vocab, **txt_kwargs)]
    if test: datasets.append(TextDataset.from_tokens(path, test, vocab=train_ds.vocab, **txt_kwargs))
    return data_func(datasets, path, **kwargs)

def data_from_textcsv(path:PathOrStr, tokenizer:Tokenizer, train:str='train', valid:str='valid', test:Optional[str]=None,
                      data_func:DataFunc=standard_data, vocab:Vocab=None, **kwargs) -> DataBunch:
    "Creates a `DataBunch` from texts in csv files."
    path=Path(path)
    txt_kwargs, kwargs = extract_kwargs(['max_vocab', 'chunksize', 'min_freq', 'n_labels'], kwargs)
    train_ds = TextDataset.from_csv(path, tokenizer, train, vocab=vocab, **txt_kwargs)
    datasets = [train_ds, TextDataset.from_csv(path, tokenizer, valid, vocab=train_ds.vocab, **txt_kwargs)]
    if test: datasets.append(TextDataset.from_csv(path, tokenizer, test, vocab=train_ds.vocab, **txt_kwargs))
    return data_func(datasets, path, **kwargs)

def data_from_textfolder(path:PathOrStr, tokenizer:Tokenizer, train:str='train', valid:str='valid', test:Optional[str]=None,
                         shuffle:bool=True, data_func:DataFunc=standard_data, vocab:Vocab=None, **kwargs):
    "Creates a `DataBunch` from text files in folders."
    path=Path(path)
    txt_kwargs, kwargs = extract_kwargs(['max_vocab', 'chunksize', 'min_freq', 'n_labels'], kwargs)
    train_ds = TextDataset.from_folder(path, tokenizer, train, shuffle=shuffle, vocab=vocab, **txt_kwargs)
    datasets = [train_ds, TextDataset.from_folder(path, tokenizer, valid, classes=train_ds.classes,
                                        shuffle=shuffle, vocab=train_ds.vocab, **txt_kwargs)]
    if test: datasets.append(TextDataset.from_folder(path, tokenizer, test, classes=train_ds.classes,
                                        shuffle=shuffle, vocab=train_ds.vocab, **txt_kwargs))
    return data_func(datasets, path, **kwargs)

def data_from_tabulardf(path, train_df:DataFrame, valid_df:DataFrame, dep_var:str, test_df:OptDataFrame=None,
                        tfms:OptTabTfms=None, cat_names:OptStrList=None, cont_names:OptStrList=None,
                        stats:OptStats=None, log_output:bool=False, **kwargs) -> DataBunch:
    "Creates a `DataBunch` from train/valid/test dataframes."
    train_ds = TabularDataset.from_dataframe(train_df, dep_var, tfms, cat_names, cont_names, stats, log_output)
    valid_ds = TabularDataset.from_dataframe(valid_df, dep_var, train_ds.tfms, train_ds.cat_names,
                                             train_ds.cont_names, train_ds.stats, log_output)
    datasets = [train_ds, valid_ds]
    if test_df:
        datasets.appendTabularDataset.from_dataframe(valid_df, dep_var, train_ds.tfms, train_ds.cat_names,
                                                     train_ds.cont_names, train_ds.stats, log_output)
    return DataBunch.create(*datasets, path=path, **kwargs)