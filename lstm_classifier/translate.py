from __future__ import division

import onmt
import torch
import argparse
import math
import codecs
import sys
import csv

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-num_classes', default=2, type=int,
                    help="""Number of classes""")
parser.add_argument('-src',   required=True,
                    help='Source sequence to check')
parser.add_argument('-output', default=None,
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=50,
                    help='Maximum sentence length.')
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")



def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal/wordsTotal)))

def addone(f):
    for line in f:
        yield line
    yield None

def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = onmt.Translator(opt)

    src1Batch, src2Batch, tgtBatch = [], [], []

    count = 0
    total_correct, total_words, total_loss = 0, 0, 0
    outputs, predictions, sents = [], [], []

    fhandle = codecs.open(opt.src, "r", "utf-8")
    tsv_reader = csv.reader(fhandle, delimiter='\t')
    #tgtF = open(opt.tgt, "r")
    for line in tsv_reader:
        count += 1

        sents.append(line)
        sent1 = line[1]
        sent2 = line[2]
        label = line[3]
            
        src1Tokens = sent1.split()
        src2Tokens = sent2.split()
            
        if len(src1Tokens) <= opt.max_sent_length:
            src1Batch += [src1Tokens]
        else:
            src1Batch += [src1Tokens[:opt.max_sent_length]]
            
        if len(src2Tokens) <= opt.max_sent_length:
            src2Batch += [src2Tokens]
        else:
            src2Batch += [src2Tokens[:opt.max_sent_length]]

        tgtBatch += [label]

        if len(src1Batch) < opt.batch_size:
            continue
        num_correct, batchSize, outs, preds = translator.translate(src1Batch, src2Batch, tgtBatch)
 
        total_correct += num_correct.data.item()
        total_words += batchSize
        outputs += outs.data.tolist()
        predictions += preds.data.tolist()

        src1Batch, src2Batch, tgtBatch = [], [], []
        if count%1000 == 0:
            print('Completed: ', str(count))
            sys.stdout.flush()
    if len(src1Batch) != 0:
        num_correct, batchSize, outs, preds = translator.translate(src1Batch, src2Batch, tgtBatch)
        total_correct += num_correct.data.item()
        total_words += batchSize
        outputs += outs.data.tolist()
        predictions += preds.data.tolist()

    print(len(sents), len(outputs), len(predictions))
    if opt.output:
        with codecs.open(opt.output, "w", "utf-8") as outF:
            tsv_writer = csv.writer(outF, delimiter='\t')
            for i in range(len(sents)):
                tsv_writer.writerow(
                    [sents[i][0], sents[i][1], sents[i][2], \
                     sents[i][3], sents[i][4], sents[i][5], \
                     outputs[i][0], outputs[i][1], predictions[i]])

    print('Accuracy: ', str((total_correct*100)/total_words))
    print('')


if __name__ == "__main__":
    main()
