# Create a data directory in the lstm_classifier folder
mkdir data

### Run the NIPS dataset ###

# Preprocess the NIPS data for L-TSort model. 
# The input files are the outcomes of prepare_data script
# Download the glove 840B embedding from 
# https://nlp.stanford.edu/projects/glove/
python preprocess.py -train_src ../nips_data/train.tsv -valid_src ../nips_data/dev.tsv -save_data data/nips_data -src_vocab_size 16721 -src_embedding PATH_TO/glove.840B.300d.txt -seq_length 100

# Train the model
python train.py -data data/nips_data.train.pt -brnn -gpus 1 -sequence_length 105 -pre_word_vecs_enc data/nips_data.embed.pt -save_model nips_lstm/model

# Run the best model on test set
# This script creates a nips_lstm_results.tsv file. 
# This is later useful to get the sentence orders 
# for the document using topological sort script.
python translate.py -model nips_lstm.pt -src ../nips_data/test.tsv -max_sent_length 100 -gpu 0 -output nips_lstm_results.tsv

# Run topological sort to get predicted orders and the metrics
python topological_sort.py --file_path nips_lstm_results.tsv


### Run the SIND dataset ###

# Preprocess the SIND data for L-TSort model. 
# The input files are the outcomes of prepare_data script
# Download the glove 840B embedding from 
# https://nlp.stanford.edu/projects/glove/
python preprocess.py -train_src ../sind_data/train.tsv -valid_src ../sind_data/dev.tsv -save_data data/sind_data -src_vocab_size 30861 -src_embedding PATH_TO/glove.840B.300d.txt -seq_length 50

# Train the model
python train.py -data data/sind_data.train.pt -brnn -gpus 1 -sequence_length 105 -pre_word_vecs_enc data/sind_data.embed.pt -save_model sind_lstm/model

# Run the best model on test set
python translate.py -model sind_lstm.pt -src ../sind_data/test.tsv -max_sent_length 50 -gpu 0 -output sind_lstm_results.tsv

# Run topological sort to get predicted orders and the metrics
python topological_sort.py --file_path sind_lstm_results.tsv