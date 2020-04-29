import torch
import csv
import json
import random
import argparse

from transformers import (WEIGHTS_NAME, BertConfig,
                BertForSequenceClassification, BertTokenizer)

class DataHandler:    
    
    def __init__(self, directory, task_name):
        for _, tokenizer_class, pretrained_weights in \
        [(BertForSequenceClassification, 
        BertTokenizer, 
        'bert-base-uncased')]:
            self.tokenizer = tokenizer_class.from_pretrained(
                                            pretrained_weights)
        self.directory = directory
        self.task = task_name
            
    def get_filenames(self, split):
        with open(split, "r") as inp:
            filenames = inp.read()
        filenames = filenames.split('\n')[:-1]
        return filenames

    def load_json_file(self, split):
        data = json.load(
            open(self.directory + split + '.story-in-sequence.json','r'))
        return data

    def get_story_text(self, data):
        story_sentences = {}
        annotations = data['annotations']
        for annotation in annotations:
            story_id = annotation[0]['story_id']
            story_sentences.setdefault(story_id, [])
            story_sentences[story_id].append(annotation[0]['original_text'])
        return story_sentences
    
    def truncate_test(self, sent1, sent2):
        s1i = self.tokenizer.encode(sent1)
        s2i = self.tokenizer.encode(sent2)
        if len(s1i) < 50:
            sent2 = self.tokenizer.decode(s2i[:100-len(s1i)])
        elif len(s2i) < 50:
            sent1 = self.tokenizer.decode(s1i[:100-len(s2i)])
        else:
            sent1 = self.tokenizer.decode(s1i[:50])
            sent2 = self.tokenizer.decode(s2i[:50])
        inp = self.tokenizer.encode(sent1, sent2, add_special_tokens=True)
        assert len(inp) < 105
        return sent1, sent2
               
    def write_test(self, split, filename, out_dir):
        dpath = self.directory + 'split/' + split
        filenames = self.get_filenames(dpath)
        
        x, y = 0, 0
        filename = out_dir + filename
        with open(filename, "w") as out:
            tsv_writer = csv.writer(out, delimiter='\t')
            
            for file in filenames:
                if self.task == 'nips':
                    with open(self.directory + 'txt_tokenized/' + 'a' + file + '.txt', 
                    'r') as inp:
                        lines = inp.readlines()
                else:
                    with open(
                        self.directory + 'txt_tokenized/' + file, 'r') as inp:
                        lines = inp.readlines()
                lines = [line.strip() for line in lines]
                y += 1
                
                if y%100 == 0:
                    print(y, x)
                    #break
                    
                tmp = []
                for i in range(len(lines)):
                
                    for j in range(i+1, len(lines)):                  
                    
                        sent1 = lines[i].lower()
                        sent2 = lines[j].lower()

                        #check if tokenized input is greater than 100
                        inputs = self.tokenizer.encode(
                                                sent1, 
                                                sent2, 
                                                add_special_tokens=True)
                        length = len(inputs)
                        if length > 100:
                            #print(sent1, sent2)
                            sent1, sent2 = self.truncate_test(
                                                        sent1, sent2)
                            #print(sent1, sent2)

                        x += 1
                        r = random.random()
                        if r >= 0.5:
                            tmp.append([str(y)+'-'+str(len(lines)), \
                                                sent1, sent2, 1, i, j])
                        else:
                            tmp.append([str(y)+'-'+str(len(lines)), \
                                                sent2, sent1, 0, j, i])

                for row in tmp:
                    #adding no of pairs of sentences in the end
                    row[0] += '-' + str(len(tmp))
                    tsv_writer.writerow(row)

    def write_test_sind(self, split, filename, out_dir):
        data = self.load_json_file(split)
        story_sentences = self.get_story_text(data)
 
        x, y = 0, 0
        filename = out_dir + filename
        with open(filename, "w") as out:
            tsv_writer = csv.writer(out, delimiter='\t')
            for story_id in story_sentences.keys():
                y += 1                
                if y%100 == 0:
                    print(y, x) 

                story = story_sentences[story_id]
                tmp = []
                for i in range(len(story)):
                    for j in range(i+1, len(story)):

                        sent1 = story[i]
                        sent2 = story[j]
                        #check if tokenized input is greater than 100
                        inputs = self.tokenizer.encode(
                                            sent1.lower(), 
                                            sent2.lower(), 
                                            add_special_tokens=True)

                        length = len(inputs)
                        if length > 100:
                            #print(sent1, sent2)
                            sent1, sent2 = self.truncate_test(
                                                        sent1, sent2)
                            #print(sent1, sent2)

                        x += 1
                        r = random.random()
                        if r >= 0.5:
                            tmp.append([str(y)+'-'+str(len(story)), \
                                                sent1, sent2, 1, i, j])
                        else:
                            tmp.append([str(y)+'-'+str(len(story)), \
                                                sent2, sent1, 0, j, i])

                for row in tmp:
                    #adding no of pairs of sentences in the end
                    row[0] += '-' + str(len(tmp))
                    tsv_writer.writerow(row)
    
    def get_convert_write(self, split, filename, out_dir):
        dpath = self.directory + 'split/' + split
        filenames = self.get_filenames(dpath)
        
        x, y = 0, 0
        filename = out_dir + filename
        with open(filename, "w") as out:
            tsv_writer = csv.writer(out, delimiter='\t')
            
            for file in filenames:
                if self.task == 'nips':
                    with open(
                        self.directory + 'txt_tokenized/' + 'a' + file + '.txt', 
                        'r') as inp:
                        lines = inp.readlines()
                else:
                    with open(
                        self.directory + 'txt_tokenized/' + file, 'r') as inp:
                        lines = inp.readlines()
                lines = [line.strip() for line in lines]
                y += 1
                
                if y%100 == 0:
                    print(y, x)               
                    
                for i in range(len(lines)):
                
                    for j in range(i+1, len(lines)):                  
                    
                        sent1 = lines[i]
                        sent2 = lines[j]

                        #check if tokenized input is greater than 100
                        inputs = self.tokenizer.encode(
                                            sent1.lower(), 
                                            sent2.lower(), 
                                            add_special_tokens=True)
                        if len(inputs) > 100:
                            continue

                        x += 1
                        tsv_writer.writerow(
                            [split+'-'+str(y)+'-'+str(x), sent1, sent2, 1])
                        tsv_writer.writerow(
                            [split+'-'+str(y)+'-'+str(x), sent2, sent1, 0])

    def get_convert_write_sind(self, split, filename, out_dir):
        data = self.load_json_file(split)
        story_sentences = self.get_story_text(data)

        x, y = 0, 0
        filename = out_dir + filename
        with open(filename, "w") as out:
            tsv_writer = csv.writer(out, delimiter='\t')
            for story_id in story_sentences.keys():
                y += 1                
                if y%100 == 0:
                    print(y, x) 

                story = story_sentences[story_id]
                for i in range(len(story)):
                    for j in range(i+1, len(story)):

                        sent1 = story[i]
                        sent2 = story[j]
                        #check if tokenized input is greater than 100
                        inputs = self.tokenizer.encode(
                                            sent1.lower(), 
                                            sent2.lower(), 
                                            add_special_tokens=True)
                        if len(inputs) > 100:
                            continue

                        x += 1
                        tsv_writer.writerow(
                            [split+'-'+str(y)+'-'+str(x), sent1, sent2, 1])
                        tsv_writer.writerow(
                            [split+'-'+str(y)+'-'+str(x), sent2, sent1, 0])

        
def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str,
                         required=True, help="The input data dir.")
    parser.add_argument("--out_dir", default='', type=str,
                         help="The dir to save the output files.")
    parser.add_argument("--task_name", default='', type=str, required=True,
                         help="Task name can be nips | nsf | aan | sind")
    args = parser.parse_args()

    handler = DataHandler(args.data_dir, args.task_name)
    if args.task_name == 'nips':
        handler.get_convert_write('2013le_papers', 'train.tsv', args.out_dir)
        handler.get_convert_write('2014_papers', 'dev.tsv', args.out_dir)
        handler.write_test('2015_papers', 'test.tsv', args.out_dir)
    elif args.task_name == 'sind':
        handler.get_convert_write_sind('train', 'train.tsv', args.out_dir)
        handler.get_convert_write_sind('val', 'dev.tsv', args.out_dir)
        handler.write_test_sind('test', 'test.tsv', args.out_dir)
    else:
        handler.get_convert_write('train', 'train.tsv', args.out_dir)
        handler.get_convert_write('valid', 'dev.tsv', args.out_dir)
        handler.write_test('test', 'test.tsv', args.out_dir)
    
if __name__ == "__main__":
    main()
