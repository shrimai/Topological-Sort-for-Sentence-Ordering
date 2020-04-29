import onmt
import numpy as np
import argparse
import torch
import codecs
import json
import sys
import csv

parser = argparse.ArgumentParser(description='preprocess.py')

##
## **Preprocess Options**
##

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=20000,
                    help="Size of the source vocabulary")
parser.add_argument('-src_vocab', default=None,
                    help="Path to an existing source vocabulary")
parser.add_argument('-src_embedding', default=None,
                    help="Path to an existing source embedding matrix")

parser.add_argument('-seq_length', type=int, default=100,
                    help="Maximum sequence length")
parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()
print(opt)

torch.manual_seed(opt.seed)

def makeVocabulary(filename, size):
    vocab = onmt.Dict(
        [onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD, \
        onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD], lower=opt.lower, seq_len=opt.seq_length)

    count = 0
    with codecs.open(filename, "r", "utf-8") as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for line in tsv_reader:
            sent = line[1]
            for word in sent.split():
                vocab.add(word)
            sent = line[2]
            for word in sent.split():
                vocab.add(word)
            count += 1

    with codecs.open(opt.valid_src, "r", "utf-8") as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for line in tsv_reader:
            sent = line[1]
            for word in sent.split():
                vocab.add(word)
            sent = line[2]
            for word in sent.split():
                vocab.add(word)

    fname = opt.valid_src.split('.tsv')[0][:-3] + 'test.tsv'
    with codecs.open(fname, "r", "utf-8") as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for line in tsv_reader:
            sent = line[1]
            for word in sent.split():
                vocab.add(word)
            sent = line[2]
            for word in sent.split():
                vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize):

    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading '+name+' vocabulary from \''+vocabFile + '\'...')
        vocab = onmt.Dict()
        vocab.lower = opt.lower
        vocab.seq_length = opt.seq_length
        vocab.loadFile(vocabFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFile, vocabSize)

        vocab = genWordVocab

    print()
    return vocab

def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)

def createEmbedMatrix(srcDicts):

    print('Creating Embed matrix ...')
    src_embed = torch.FloatTensor(torch.randn(srcDicts.size(), 300))
    found = 0
    f = codecs.open(opt.src_embedding, 'rb', 'utf-8')
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        idx = srcDicts.lookup(word)
        if idx:
            src_embed[idx] = torch.from_numpy(embedding)
            found += 1

    print('No of words from the vocab in the Glove: ' + str(found))

    return src_embed

def makeData(srcFile, srcDicts):
    src1, src2, tgt = [], [], []
    sizes1, sizes2 = [], []
    count, ignored = 0, 0

    print('Processing %s ...' % (srcFile))

    with codecs.open(srcFile, "r", "utf-8") as srcF:
        tsv_reader = csv.reader(srcF, delimiter='\t')
        for line in tsv_reader:
            sent1 = line[1]
            sent2 = line[2]
            label = line[3]
        
            src1Words = sent1.split()
            src2Words = sent2.split()

            if len(src1Words) > opt.seq_length:
                src1Words = src1Words[:opt.seq_length]
            if len(src2Words) > opt.seq_length:
                src2Words = src2Words[:opt.seq_length]

            src1 += [srcDicts.convertToIdx(src1Words,
                                onmt.Constants.UNK_WORD, padding=True)]
            src2 += [srcDicts.convertToIdx(src2Words,
                                onmt.Constants.UNK_WORD, padding=True)]
            tgt += [torch.LongTensor([int(label)])]
            sizes1 += [len(src1Words)]
            sizes2 += [len(src2Words)]
            count += 1

            if count % opt.report_every == 0:
                print('... %d sentences prepared' % count)

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src1))
        src1 = [src1[idx] for idx in perm]
        src2 = [src2[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes1 = [sizes1[idx] for idx in perm]
        sizes2 = [sizes2[idx] for idx in perm]

    print('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
          (len(src1), ignored, opt.seq_length))

    return src1, src2, tgt

def main():

    dicts = {}
    print('Preparing source vocab ....')
    dicts['src'] = initVocabulary(
                            'source', 
                            opt.train_src, 
                            opt.src_vocab,
                            opt.src_vocab_size
                            )
    if opt.src_embedding:
        embedding = createEmbedMatrix(dicts['src'])
        torch.save(embedding, opt.save_data + '.embed.pt')

    print('Preparing training ...')
    train = {}
    train['src1'], train['src2'], train['tgt'] = makeData(
                                            opt.train_src,
                                            dicts['src']
                                            )

    print('Preparing validation ...')
    valid = {}
    valid['src1'], valid['src2'], valid['tgt'] = makeData(
                                    opt.valid_src,
                                    dicts['src']
                                    )

    if opt.src_vocab is None:
        saveVocabulary(
            'source', 
            dicts['src'], 
            opt.save_data + 
            '.src.dict'
            )


    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': train,
                 'valid': valid,
                }
    torch.save(save_data, opt.save_data + '.train.pt')


if __name__ == "__main__":
    main()
