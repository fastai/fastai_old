from nb_002 import *

from dataclasses import dataclass
from typing import Any, Collection, Callable


def dict_groupby(iterable, key=None):
    return {k:list(v) for k,v in itertools.groupby(sorted(iterable, key=key), key=key)}

def resolve_pipeline(tfms, **kwargs):
    if len(tfms)==0: return noop
    grouped_tfms = dict_groupby(tfms, lambda o: o.__annotations__['return'])
    lighting_tfms,coord_tfms,affine_tfms,pixel_tfms,final_tfms = [
        resolve_tfms(grouped_tfms.get(o)) for o in TfmType]
    lighting_tfm = apply_lighting_tfm(compose(lighting_tfms))
    affine_tfm = apply_affine_tfm(affine_tfms, func=compose(coord_tfms), **kwargs)
    final_tfm = compose(final_tfms)
    pixel_tfm = compose(pixel_tfms)
    return lambda x,**k: final_tfm(affine_tfm(lighting_tfm(pixel_tfm(x)), **k))

@reg_transform
def pad(x, padding, mode='reflect') -> TfmType.Pixel:
    return F.pad(x[None], (padding,)*4, mode=mode)[0]

@reg_transform
def crop(x, size, row_pct:uniform, col_pct:uniform) -> TfmType.Final:
    size = listify(size,2)
    rows,cols = size
    row = int((x.size(1)-rows)*row_pct)
    col = int((x.size(2)-cols)*col_pct)
    return x[:, row:row+rows, col:col+cols]

@dataclass
class TfmDataset(Dataset):
    ds: Dataset
    tfms: Collection[Callable]

    def __len__(self): return len(self.ds)

    def __getitem__(self,idx):
        x,y = self.ds[idx]
        return resolve_pipeline(tfms)(x),y


