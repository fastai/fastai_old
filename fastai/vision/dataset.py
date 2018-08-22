from .imports import *


def find_classes(folder:Path) -> Collection[Path]:
    "Find all the classes in a given folder"
    classes = [d for d in folder.iterdir()
               if d.is_dir() and not d.name.startswith('.')]
    assert(len(classes)>0)
    return sorted(classes, key=lambda d: d.name)

def get_image_files(c:Path) -> Collection[Path]:
    "Find all the image files in a given foldder"
    return [o for o in list(iterdir())
            if not o.name.startswith('.') and not o.is_dir()]

def pil2tensor(image:Image) -> Tensor:
    "Transforms a PIL Image in a torch tensor"
    arr = ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    arr = arr.view(image.size[1], image.size[0], -1)
    arr = arr.permute(2,0,1)
    return arr.float().div_(255)

class FilesDataset(Dataset):
    "Basic class to create a dataset from folders and files"

    def __init__(self, fns:Collection[FileLike], labels:Collection[str], classes:Collection[str]=None):
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

