import numpy as np
import onmt
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

class LSTMEncoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(LSTMEncoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)
        self.rnn = nn.LSTM(input_size, self.hidden_size,
                        num_layers=opt.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden=None):
        if isinstance(input, tuple):
            emb = pack(self.word_lut(input[0]), input[1])
        else:
            emb = self.word_lut(input)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        return outputs, hidden_t

class LSTMClassifier(nn.Module):

    def __init__(self, opt, dicts):
        super(LSTMClassifier, self).__init__()
        self.encoder = LSTMEncoder(opt, dicts)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(opt.rnn_size * 2, opt.rnn_size)
        self.classifier = nn.Linear(opt.rnn_size, opt.num_classes)

    def load_pretrained_vectors(self, opt):
        self.encoder.load_pretrained_vectors(opt)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            h = h.view(h.size(0) // 2, 2, h.size(1), h.size(2))
            h = h.transpose(1, 2).contiguous() 
            h = h.view(h.size(0), h.size(1), 2 * h.size(3))
            return h
        else:
            return h

    def forward(self, input):
        
        src1 = input[0]
        src2 = input[1]
        context1, enc_hidden1 = self.encoder(src1[0])
        context2, enc_hidden2 = self.encoder(src2[0])

        ht1 = self._fix_enc_hidden(enc_hidden1[0])[-1]
        ht2 = self._fix_enc_hidden(enc_hidden2[0])[-1]
        # concat hidden1 and hidden2
        enc_hidden = torch.cat((ht1, ht2), 1)

        lin_out = torch.tanh(self.dropout(self.linear(enc_hidden)))
        logits = self.classifier(lin_out)
        return logits