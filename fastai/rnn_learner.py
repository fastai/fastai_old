from .train import *
from .models.rnn import get_language_model, get_rnn_classifier
from .metrics import accuracy

def convert_weights(wgts:Weights, stoi_wgts:Dict[str,int], itos_new:Collection[str]) -> Weights:
    "Converts the model weights to go with a new vocabulary."
    dec_bias, enc_wgts = wgts['1.decoder.bias'], wgts['0.encoder.weight']
    bias_m, wgts_m = dec_bias.mean(0), enc_wgts.mean(0)
    new_w = enc_wgts.new_zeros((len(itos_new),enc_wgts.size(1))).zero_()
    new_b = dec_bias.new_zeros((len(itos_new),)).zero_()
    for i,w in enumerate(itos_new):
        r = stoi_wgts[w] if w in stoi_wgts else -1
        new_w[i] = enc_wgts[r] if r>=0 else wgts_m
        new_b[i] = dec_bias[r] if r>=0 else bias_m
    wgts['0.encoder.weight'] = new_w
    wgts['0.encoder_dp.emb.weight'] = new_w.clone()
    wgts['1.decoder.weight'] = new_w.clone()
    wgts['1.decoder.bias'] = new_b
    return wgts

def lm_split(model:Model) -> List[Model]:
    "Splits a RNN model in groups for differential learning rates."
    groups = [nn.Sequential(rnn, dp) for rnn, dp in zip(model[0].rnns, model[0].hidden_dps)]
    groups.append(nn.Sequential(model[0].encoder, model[0].encoder_dp, model[1]))
    return groups


def rnn_classifier_split(model:Model) -> List[Model]:
    "Splits a RNN model in groups."
    groups = [nn.Sequential(model[0].encoder, model[0].encoder_dp)]
    groups += [nn.Sequential(rnn, dp) for rnn, dp in zip(model[0].rnns, model[0].hidden_dps)]
    groups.append(model[1])
    return groups


class RNNLearner(Learner):
    "Basic class for a Learner in RNN"
    def __init__(self, data:DataBunch, model:Model, bptt:int=70, split_func:OptSplitFunc=None, clip:float=None,
                 adjust:bool=False, alpha:float=2., beta:float=1., **kwargs):
        super().__init__(data, model)
        self.callbacks.append(RNNTrainer(self, bptt, alpha=alpha, beta=beta, adjust=adjust))
        if clip: self.callback_fns.append(partial(GradientClipping, clip=clip))
        if split_func: self.split(split_func)
        self.metrics = [accuracy]

    def save_encoder(self, name:str):
        "Saves the encoder to the model directory"
        torch.save(self.model[0].state_dict(), self.path/self.model_dir/f'{name}.pth')

    def load_encoder(self, name:str):
        "Loads the encoder from the model directory"
        self.model[0].load_state_dict(torch.load(self.path/self.model_dir/f'{name}.pth'))

    def load_pretrained(self, wgts_fname:str, itos_fname:str):
        "Loads a pretrained model and adapts it to the data vocabulary."
        old_itos = pickle.load(open(self.path/self.model_dir/f'{itos_fname}.pkl', 'rb'))
        old_stoi = {v:k for k,v in enumerate(old_itos)}
        wgts = torch.load(self.path/self.model_dir/f'{wgts_fname}.pth', map_location=lambda storage, loc: storage)
        wgts = convert_weights(wgts, old_stoi, self.data.train_ds.vocab.itos)
        self.model.load_state_dict(wgts)

    @classmethod
    def language_model(cls, data:DataBunch, bptt:int=70, emb_sz:int=400, nh:int=1150, nl:int=3, pad_token:int=1,
                       drop_mult:float=1., tie_weights:bool=True, bias:bool=True, qrnn:bool=False,
                       pretrained_fnames:OptStrTuple=None, **kwargs) -> 'RNNLearner':
        "Creates a `Learner` with a language model."
        dps = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * drop_mult
        vocab_size = len(data.train_ds.vocab.itos)
        model = get_language_model(vocab_size, emb_sz, nh, nl, pad_token, input_p=dps[0], output_p=dps[1],
                    weight_p=dps[2], embed_p=dps[3], hidden_p=dps[4], tie_weights=tie_weights, bias=bias, qrnn=qrnn)
        learn = cls(data, model, bptt, split_func=lm_split, **kwargs)
        if pretrained_fnames is not None: learn.load_pretrained(*pretrained_fnames)
        return learn

    @classmethod
    def classifier(cls, data:DataBunch, bptt:int=70, max_len:int=70*20, emb_sz:int=400, nh:int=1150, nl:int=3,
                   layers:Collection[int]=None, drops:Collection[float]=None, pad_token:int=1,
                   drop_mult:float=1., qrnn:bool=False, **kwargs) -> 'RNNLearner':
        "Creates a RNN classifier."
        dps = np.array([0.4,0.5,0.05,0.3,0.4]) * drop_mult
        if layers is None: layers = [50]
        if drops is None:  drops = [0.1]
        vocab_size = len(data.train_ds.vocab.itos)
        n_class = len(data.train_ds.classes)
        layers = [emb_sz*3] + layers + [n_class]
        drops = [dps[4]] + drops
        model = get_rnn_classifier(bptt, max_len, n_class, vocab_size, emb_sz, nh, nl, pad_token,
                    layers, drops, input_p=dps[0], weight_p=dps[1], embed_p=dps[2], hidden_p=dps[3], qrnn=qrnn)
        learn = cls(data, model, bptt, split_func=rnn_classifier_split, **kwargs)
        return learn