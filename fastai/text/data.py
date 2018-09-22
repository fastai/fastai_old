from ..torch_core import *
from .transform import *

TextMtd = IntEnum('TextMtd', 'CSV TOK IDS')

class TextDataset():
    "Basic dataset for NLP tasks."

    def __init__(self, path:PathOrStr, tokenizer:Tokenizer, vocab:Vocab=None, max_vocab:int=60000, chunksize:int=10000,
                 name:str='train', min_freq:int=2, n_labels:int=1, create_mtd:TextMtd=TextMtd.CSV, classes:Classes=None):
        self.path,self.tokenizer,self.max_vocab,self.min_freq = Path(path/'tmp'),tokenizer,max_vocab,min_freq
        self.chunksize,self.name,self.n_labels,self.create_mtd = chunksize,name,n_labels,create_mtd
        self.vocab=vocab
        os.makedirs(self.path, exist_ok=True)
        if not self.check_toks(): self.tokenize()
        if not self.check_ids():  self.numericalize()

        if self.vocab is None: self.vocab = Vocab(self.path)
        self.ids = np.load(self.path/f'{self.name}_ids.npy')
        if os.path.isfile(self.path/f'{self.name}_lbl.npy'):
            self.labels = np.load(self.path/f'{self.name}_lbl.npy')
        else: self.labels = np.zeros((len(self.ids),), dtype=np.int64)
        self.classes = classes if classes else np.unique(self.labels)

    def __getitem__(self, idx:int) -> Tuple[int,int]: return self.ids[idx],self.labels[idx]
    def __len__(self) -> int: return len(self.ids)

    def general_check(self, pre_files:Collection[PathOrStr], post_files:Collection[PathOrStr]):
        "Checks that post_files exist and were modified after all the prefiles."
        if not np.all([os.path.isfile(fname) for fname in post_files]): return False
        for pre_file in pre_files:
            if os.path.getmtime(pre_file) > os.path.getmtime(post_files[0]): return False
        return True

    def check_ids(self) -> bool:
        "Checks if a new numericalization is needed."
        if self.create_mtd >= TextMtd.IDS: return True
        if not self.general_check([self.tok_files[0],self.id_files[1]], self.id_files): return False
        itos = pickle.load(open(self.id_files[1], 'rb'))
        h = hashlib.sha1(np.array(itos))
        with open(self.id_files[2]) as f:
            if h.hexdigest() != f.read() or len(itos) > self.max_vocab + 2: return False
        toks,ids = np.load(self.tok_files[0]),np.load(self.id_files[0])
        if len(toks) != len(ids): return False
        return True

    def check_toks(self) -> bool:
        "Checks if a new tokenization is needed."
        if self.create_mtd >= TextMtd.TOK: return True
        if not self.general_check([self.csv_file], self.tok_files): return False
        with open(self.tok_files[1]) as f:
            if repr(self.tokenizer) != f.read(): return False
        return True

    def tokenize(self):
        "Tokenizes the texts in the csv file"
        print(f'Tokenizing {self.name}. This might take a while so you should grab a coffee.')
        curr_len = get_chunk_length(self.csv_file, self.chunksize)
        dfs = pd.read_csv(self.csv_file, header=None, chunksize=self.chunksize)
        tokens,labels = [],[]
        for _ in progress_bar(range(curr_len), leave=False):
            df = next(dfs)
            lbls = df.iloc[:,range(self.n_labels)].values.astype(np.int64)
            texts = f'\n{BOS} {FLD} 1 ' + df[self.n_labels].astype(str)
            for i in range(self.n_labels+1, len(df.columns)):
                texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
            toks = self.tokenizer.process_all(texts)
            tokens += toks
            labels += list(np.squeeze(lbls))
        np.save(self.tok_files[0], np.array(tokens))
        np.save(self.path/f'{self.name}_lbl.npy', np.array(labels))
        with open(self.tok_files[1],'w') as f: f.write(repr(self.tokenizer))

    def numericalize(self):
        "Numericalizes the tokens in the token file"
        print(f'Numericalizing {self.name}.')
        toks = np.load(self.tok_files[0])
        if self.vocab is None: self.vocab = Vocab.create(self.path, toks, self.max_vocab, self.min_freq)
        ids = np.array([self.vocab.numericalize(t) for t in toks])
        np.save(self.id_files[0], ids)

    def clear(self):
        "Removes temporary files"
        files = [self.path/f'{self.name}_{suff}.npy' for suff in ['ids','tok','lbl']]
        files.append(self.path/f'{self.name}.csv')
        for file in files:
            if os.path.isfile(file): os.remove(file)

    @property
    def csv_file(self) -> Path: return self.path/f'{self.name}.csv'
    @property
    def tok_files(self) -> List[Path]: return [self.path/f'{self.name}_tok.npy', self.path/'tokenize.log']
    @property
    def id_files(self) -> List[Path]:
        return [self.path/f'{self.name}_ids.npy', self.path/'itos.pkl', self.path/'numericalize.log']

    @classmethod
    def from_ids(cls, folder:PathOrStr, name:str, id_suff:str='_ids', lbl_suff:str='_lbl',
                 itos:str='itos.pkl', **kwargs) -> 'TextDataset':
        "Creates a dataset from an id, a dictionary and label file."
        orig = [Path(folder/file) for file in [f'{name}{id_suff}.npy', f'{name}{lbl_suff}.npy', itos]]
        dest = [Path(folder)/'tmp'/file for file in [f'{name}_ids.npy', f'{name}_lbl.npy', 'itos.pkl']]
        maybe_copy(orig, dest)
        return cls(folder, None, name=name, create_mtd=TextMtd.IDS, **kwargs)

    @classmethod
    def from_tokens(cls, folder:PathOrStr, name:str, tok_suff:str='_tok', lbl_suff:str='_lbl',
                    **kwargs) -> 'TextDataset':
        "Creates a dataset from a token and label file."
        orig = [Path(folder/file) for file in [f'{name}{tok_suff}.npy', f'{name}{lbl_suff}.npy']]
        dest = [Path(folder)/'tmp'/file for file in [f'{name}_tok.npy', f'{name}_lbl.npy']]
        maybe_copy(orig, dest)
        return cls(folder, None, name=name, create_mtd=TextMtd.TOK, **kwargs)

    @classmethod
    def from_csv(cls, folder:PathOrStr, tokenizer:Tokenizer, name:str, **kwargs) -> 'TextDataset':
        "Creates a dataset from texts in a csv file."
        orig = [Path(folder)/f'{name}.csv']
        dest = [Path(folder)/'tmp'/f'{name}.csv']
        maybe_copy(orig, dest)
        return cls(folder, tokenizer, name=name, **kwargs)

    @classmethod
    def from_folder(cls, folder:PathOrStr, tokenizer:Tokenizer, name:str, classes:Classes=None,
                    shuffle:bool=True, **kwargs) -> 'TextDataset':
        "Creates a dataset from a folder"
        path = Path(folder)/'tmp'
        os.makedirs(path, exist_ok=True)
        if classes is None: classes = [cls.name for cls in find_classes(Path(folder)/name)]
        texts,labels = [],[]
        for idx,label in enumerate(classes):
            for fname in (Path(folder)/name/label).glob('*.*'):
                texts.append(fname.open('r', encoding='utf8').read())
                labels.append(idx)
        texts,labels = np.array(texts),np.array(labels)
        if shuffle:
            idx = np.random.permutation(len(texts))
            texts,labels = texts[idx],labels[idx]
        df = pd.DataFrame({'text':texts, 'labels':labels}, columns=['labels','text'])
        if os.path.isfile(path/f'{name}.csv'):
            if get_total_length(path/f'{name}.csv', 10000) != len(df):
                df.to_csv(path/f'{name}.csv', index=False, header=False)
        else: df.to_csv(path/f'{name}.csv', index=False, header=False)
        return cls(folder, tokenizer, name=name, classes=classes, **kwargs)

class LanguageModelLoader():
    "Creates a dataloader with bptt slightly changing."
    def __init__(self, dataset:TextDataset, bs:int=64, bptt:int=70, backwards:bool=False):
        self.dataset,self.bs,self.bptt,self.backwards = dataset,bs,bptt,backwards
        self.data = self.batchify(np.concatenate(dataset.ids))
        self.first,self.i,self.iter = True,0,0
        self.n = len(self.data)

    def __iter__(self):
        self.i,self.iter = 0,0
        while self.i < self.n-1 and self.iter<len(self):
            if self.first and self.i == 0: self.first,seq_len = False,self.bptt + 25
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res

    def __len__(self) -> int: return (self.n-1) // self.bptt

    def batchify(self, data:np.ndarray) -> LongTensor:
        "Splits the corpus in batches."
        nb = data.shape[0] // self.bs
        data = np.array(data[:nb*self.bs]).reshape(self.bs, -1).T
        if self.backwards: data=data[::-1]
        return LongTensor(data)

    def get_batch(self, i:int, seq_len:int) -> Tuple[LongTensor, LongTensor]:
        "Creates a batch of a given seq_len"
        seq_len = min(seq_len, len(self.data) - 1 - i)
        return self.data[i:i+seq_len], self.data[i+1:i+1+seq_len].contiguous().view(-1)

class SortSampler(Sampler):
    "Go through the text data by order of length"

    def __init__(self, data_source:NPArrayList, key:KeyFunc): self.data_source,self.key = data_source,key
    def __len__(self) -> int: return len(self.data_source)
    def __iter__(self):
        return iter(sorted(range(len(self.data_source)), key=self.key, reverse=True))


class SortishSampler(Sampler):
    "Go through the text data by order of length with a bit of randomness"

    def __init__(self, data_source:NPArrayList, key:KeyFunc, bs:int):
        self.data_source,self.key,self.bs = data_source,key,bs

    def __len__(self) -> int: return len(self.data_source)

    def __iter__(self):
        idxs = np.random.permutation(len(self.data_source))
        sz = self.bs*50
        ck_idx = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([sorted(s, key=self.key, reverse=True) for s in ck_idx])
        sz = self.bs
        ck_idx = [sort_idx[i:i+sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
        ck_idx[0],ck_idx[max_ck] = ck_idx[max_ck],ck_idx[0]     # then make sure it goes first.
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:]))
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)

def pad_collate(samples:BatchSamples, pad_idx:int=1, pad_first:bool=True) -> Tuple[LongTensor, LongTensor]:
    "Function that collect samples and adds padding"
    max_len = max([len(s[0]) for s in samples])
    res = torch.zeros(max_len, len(samples)).long() + pad_idx
    for i,s in enumerate(samples): res[-len(s[0]):,i] = LongTensor(s[0])
    return res, LongTensor([s[1] for s in samples]).squeeze()