import onmt
import torch.nn as nn
import torch
from torch.autograd import Variable


class Translator(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        checkpoint = torch.load(opt.model)
        model_opt = checkpoint['opt']
        self.src_dict = checkpoint['dicts']['src']

        model = onmt.Models.LSTMClassifier(model_opt, self.src_dict)
        model.load_state_dict(checkpoint['model'])

        if opt.cuda:
            model.cuda()
            self.gpu = True
        else:
            model.cpu()
            self.gpu = False

        self.model = model
        self.model.eval()

    def buildData(self, src1Batch, src2Batch, goldBatch):
        src1Data = [self.src_dict.convertToIdx(b,
                    onmt.Constants.UNK_WORD, padding=True) for b in src1Batch]
        src2Data = [self.src_dict.convertToIdx(b,
                    onmt.Constants.UNK_WORD, padding=True) for b in src2Batch]
        tgtData = []
        if goldBatch:
            for label in goldBatch:
                tgtData += [torch.LongTensor([int(label)])]
                

        return onmt.Dataset(src1Data, src2Data, tgtData,
            self.opt.batch_size, self.opt.cuda)

    def translateBatch(self, src1Batch, src2Batch, tgtBatch):
        batchSize = src1Batch[0].size(1)
        Batch = (src1Batch, src2Batch, tgtBatch)

        outputs = self.model(Batch)
        outputs = Variable(outputs.data, requires_grad=False, volatile=False)
        scores = nn.functional.softmax(outputs, dim=-1)
        pred = scores.max(1)[1]
        
        #if tgtBatch:
        targets = tgtBatch      
        num_correct = pred.data.eq(targets[0].data).sum()
        #else:
        #num_correct = 0
        return num_correct, batchSize, scores, pred

    def translate(self, src1Batch, src2Batch, goldBatch, flag=False):
        #  (1) convert words to indexes
        dataset = self.buildData(src1Batch, src2Batch, goldBatch)
        src1, src2, tgt = dataset[0]

        #  (2) translate
        num_correct, batchSize, outs, pred = self.translateBatch(src1, src2, tgt)

        return num_correct, batchSize, outs, pred
