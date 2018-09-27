"Module helps with formatting NLP data. Tokenizes text and creates vocab indexes"
from ..torch_core import *

__all__ = ['BaseTokenizer', 'SpacyTokenizer', 'Tokenizer', 'Vocab', 'deal_caps', 'fixup', 'replace_rep', 'replace_wrep', 
           'rm_useless_spaces', 'spec_add_spaces', 'sub_br', 'BOS', 'FLD', 'UNK', 'PAD', 'TK_UP', 'TK_REP', 'TK_REP',
           'TK_WREP', 'default_rules', 'default_spec_tok']

BOS,FLD,UNK,PAD = 'xxbos','xxfld','xxunk','xxpad'
TK_UP,TK_REP,TK_WREP = 'xxup','xxrep','xxwrep'


class BaseTokenizer():
    "Basic class for a tokenizer function."
    def __init__(self, lang:str):
        self.lang = lang

    def tokenizer(self, t:spacy.tokens.doc.Doc) -> List[str]: raise NotImplementedError
    def add_special_cases(self, toks:Collection[str]):        raise NotImplementedError

#export
class SpacyTokenizer(BaseTokenizer):
    "Little wrapper around a `spacy` tokenizer"

    def __init__(self, lang:str):
        self.tok = spacy.load(lang)

    def tokenizer(self, t:spacy.tokens.doc.Doc) -> List[str]:
        return [t.text for t in self.tok.tokenizer(t)]

    def add_special_cases(self, toks:Collection[str]):
        for w in toks:
            self.tok.tokenizer.add_special_case(w, [{ORTH: w}])

def sub_br(t:str) -> str:
    "Replaces the <br /> by \n"
    re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
    return re_br.sub("\n", t)

def spec_add_spaces(t:str) -> str:
    "Add spaces between special characters"
    return re.sub(r'([/#])', r' \1 ', t)

def rm_useless_spaces(t:str) -> str:
    "Remove multiple spaces"
    return re.sub(' {2,}', ' ', t)

def replace_rep(t:str) -> str:
    "Replace repetitions at the character level"
    def _replace_rep(m:Collection[str]) -> str:
        c,cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '
    re_rep = re.compile(r'(\S)(\1{3,})')
    return re_rep.sub(_replace_rep, t)

def replace_wrep(t:str) -> str:
    "Replace word repetitions"
    def _replace_wrep(m:Collection[str]) -> str:
        c,cc = m.groups()
        return f' {TK_WREP} {len(cc.split())+1} {c} '
    re_wrep = re.compile(r'(\b\w+\W+)(\1{3,})')
    return re_wrep.sub(_replace_wrep, t)

def deal_caps(t:str) -> str:
    "Replace words in all caps"
    res = []
    for s in re.findall(r'\w+|\W+', t):
        res += ([f' {TK_UP} ',s.lower()] if (s.isupper() and (len(s)>2)) else [s.lower()])
    return ''.join(res)

def fixup(x:str) -> str:
    "List of replacements from html strings"
    re1 = re.compile(r'  +')
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>',UNK).replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

default_rules = [fixup, replace_rep, replace_wrep, deal_caps, spec_add_spaces, rm_useless_spaces, sub_br]
default_spec_tok = [BOS, FLD, UNK, PAD]

class Tokenizer():
    "Puts together rules, a tokenizer function and a language to process text with multiprocessing"
    def __init__(self, tok_fn:Callable=SpacyTokenizer, lang:str='en', rules:ListRules=None,
                 special_cases:Collection[str]=None, n_cpus:int=None):
        self.tok_fn,self.lang,self.special_cases = tok_fn,lang,special_cases
        self.rules = rules if rules else default_rules
        self.special_cases = special_cases if special_cases else default_spec_tok
        self.n_cpus = n_cpus or num_cpus()//2

    def __repr__(self) -> str:
        res = f'Tokenizer {self.tok_fn.__name__} in {self.lang} with the following rules:\n'
        for rule in self.rules: res += f' - {rule.__name__}\n'
        return res

    def proc_text(self, t:str, tok:BaseTokenizer) -> List[str]:
        "Processes one text"
        for rule in self.rules: t = rule(t)
        return tok.tokenizer(t)

    def process_all_1(self, texts:Collection[str]) -> List[List[str]]:
        "Processes a list of texts in one process"
        tok = self.tok_fn(self.lang)
        if self.special_cases: tok.add_special_cases(self.special_cases)
        return [self.proc_text(t, tok) for t in texts]

    def process_all(self, texts:Collection[str]) -> List[List[str]]:
        "Processes a list of texts in several processes"
        if self.n_cpus <= 1: return self.process_all_1(texts)
        with ProcessPoolExecutor(self.n_cpus) as e:
            return sum(e.map(self.process_all_1, partition_by_cores(texts, self.n_cpus)), [])

class Vocab():
    "Contains the correspondance between numbers and tokens and numericalizes"

    def __init__(self, path:PathOrStr):
        self.itos = pickle.load(open(path/'itos.pkl', 'rb'))
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

    def numericalize(self, t:Collection[str]) -> List[int]:
        "Converts a list of tokens to their ids"
        return [self.stoi[w] for w in t]

    def textify(self, nums:Collection[int]) -> List[str]:
        "Converts a list of ids to their tokens"
        return ' '.join([self.itos[i] for i in nums])

    @classmethod
    def create(cls, path:PathOrStr, tokens:Tokens, max_vocab:int, min_freq:int) -> 'Vocab':
        "Create a vocabulary from a set of tokens."
        freq = Counter(p for o in tokens for p in o)
        itos = [o for o,c in freq.most_common(max_vocab) if c > min_freq]
        itos.insert(0, PAD)
        if UNK in itos: itos.remove(UNK)
        itos.insert(0, UNK)
        pickle.dump(itos, open(path/'itos.pkl', 'wb'))
        h = hashlib.sha1(np.array(itos))
        with open(path/'numericalize.log','w') as f: f.write(h.hexdigest())
        return cls(path)