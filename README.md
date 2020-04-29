# Topological Sort for Sentence Ordering
Code accompanying the paper: [Topological Sort for Sentence Ordering](https://www.cs.cmu.edu/~sprabhum/docs/Topological_Sort_for_Sentence_Ordering.pdf)

## Pre-requisites
- Python 3.6
- Pytorch 1.0.1
- [transformers](https://github.com/huggingface/transformers) 2.3.0
- [requirements](https://github.com/huggingface/transformers/blob/master/examples/requirements.txt) for using the transformers code
- [Glove](https://nlp.stanford.edu/projects/glove/) embeddings (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download) for the L-TSort model

## Data
For the AAN, NIPS and NSF data please contact the authors of [Sentence Ordering and Coherence Modeling using Recurrent Neural Networks](https://arxiv.org/pdf/1611.02654.pdf).

The SIND dataset can be downloaded from the [Visual Storytelling](http://visionandlanguage.net/VIST/dataset.html) website.

## Quickstart
Refer to example.sh file to see the commands.

- Use `prepare_data.py` script to create train, valid and test data to train classifier for AAN, NIPS, SIND and NSF datasets.
```
python prepare_data.py --data_dir nips/ --out_dir nips_data/ --task_name nips
```

### BERT-based representation (B-TSort Classifier)

The `model.py` script in `bert_classifier` should be used to train and test the B-TSort model.
- Train the B-TSort models using the following command:
```
python model.py --data_dir ../nips_data/ --output_dir ../trained_models/nips_bert/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 16
```

- Run the trained B-TSort model on the test set. This script creates a `test_results.tsv` file and saves it in the data_sir path provided. This is later useful to get the sentence orders for the document using topological sort script.
```
python model.py --data_dir ../nips_data/ --output_dir ../trained_models/nips_bert/checkpoint-X/ --do_test --per_gpu_eval_batch_size 16
```

### LSTM-based representation (L-TSort Classifier)
The code base for this classifier is in `lstm_classifier` folder.

- Create a `data` dir in the `lstm_classifier` folder and run the following preprocess command:
```
python preprocess.py -train_src ../nips_data/train.tsv -valid_src ../nips_data/dev.tsv -save_data data/nips_data -src_vocab_size 16721 -src_embedding PATH_TO/glove.840B.300d.txt -seq_length 100
```

- Train the L-TSort models using the following command:
```
python train.py -data data/nips_data.train.pt -brnn -gpus 1 -sequence_length 105 -pre_word_vecs_enc data/nips_data.embed.pt -save_model nips_lstm/model
```

- Test the L-TSort models:
```
python translate.py -model MODEL_NAME.pt -src ../nips_data/test.tsv -max_sent_length 100 -gpu 0 -output nips_lstm_results.tsv
```

### Topological Sort
Run the topological sort script on the outputs of the B-TSort and L-TSort models to calculate results for various metrics.

```
python topological_sort.py --file_path nips_data/test_results.tsv
```

## Trained Models

- Download the trained B-TSort models for each of the four datasets from the below links:
```bash
http://tts.speech.cs.cmu.edu/sentence_order/nips_bert.tar
http://tts.speech.cs.cmu.edu/sentence_order/aan_bert.tar
http://tts.speech.cs.cmu.edu/sentence_order/nsf_bert.tar
http://tts.speech.cs.cmu.edu/sentence_order/sind_bert.tar
```

- Download the trained L-TSort models for NIPS and SIND datasets from the below links:
```bash
http://tts.speech.cs.cmu.edu/sentence_order/nips_lstm.pt
http://tts.speech.cs.cmu.edu/sentence_order/sind_lstm.pt
```

## Baselines
For the code of the model denoted as AON in the paper, please email the authors of [Deep Attentive Sentence Ordering Network](https://www.aclweb.org/anthology/D18-1465/).

## Contributors
If you use this code please cite the following:

Prabhumoye, Shrimai, Ruslan Salakhutdinov, and Alan W. Black. "Topological Sort for Sentence Ordering." In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics.

    @inproceedings{topSort2020,
      title={Topological Sort for Sentence Ordering},
      author={Prabhumoye, Shrimai and Salakhutdinov, Ruslan and Black, Alan W},
      booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
      publisher={Association for Computational Linguistics},
      year={2020},
      }
