from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

import onmt


class Dataset(object):

    def __init__(self, src1Data, src2Data, tgtData, batchSize, cuda):
        self.src1 = src1Data
        self.src2 = src2Data
        if tgtData:
            self.tgt = tgtData
            assert(len(self.src1) == len(self.tgt))
        else:
            self.tgt = None
        self.cuda = cuda

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src1)/batchSize)

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(onmt.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        src1Batch, lengths1 = self._batchify(
            self.src1[index*self.batchSize:(index+1)*self.batchSize],
            align_right=False, include_lengths=True)

        src2Batch, lengths2 = self._batchify(
            self.src2[index*self.batchSize:(index+1)*self.batchSize],
            align_right=False, include_lengths=True)

        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index*self.batchSize:(index+1)*self.batchSize])
        else:
            tgtBatch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(src1Batch))
        if tgtBatch is None:
            batch = zip(indices, src1Batch, src2Batch)
        else:
            batch = zip(indices, src1Batch, src2Batch, tgtBatch)
        batch, _ = zip(*sorted(
                    zip(batch, lengths1), 
                    key=lambda x: -x[1]
                    ))
        if tgtBatch is None:
            indices, src1Batch, src2Batch = zip(*batch)
        else:
            indices, src1Batch, src2Batch, tgtBatch = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b)
            return b

        return (wrap(src1Batch), lengths1), (wrap(src2Batch), lengths2), wrap(tgtBatch)

    def __len__(self):
        return self.numBatches


    def shuffle(self):
        data = list(zip(self.src1, self.src2, self.tgt))
        self.src1, self.src2, self.tgt = zip(
                        *[data[i] for i in torch.randperm(len(data))]
                        )
