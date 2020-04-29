### NIPS dataset ###

# create a directory to save trained models
mkdir ../trained_models/nips_bert

# To train a B-TSort model
python model.py --data_dir ../nips_data/ --output_dir ../trained_models/nips_bert/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 16

# To run inference on the trained model use
# This script creates a test_results.tsv file and 
# saves it in the data_sir path provided. 
# This is later useful to get the sentence orders 
# for the document using topological sort script.
python model.py --data_dir ../nips_data/ --output_dir ../trained_models/nips_bert/ --do_test --per_gpu_eval_batch_size 16

# If you have trained your own model then use the desirable checkpoint
python model.py --data_dir ../nips_data/ --output_dir ../trained_models/nips_bert/checkpoint-X/ --do_test --per_gpu_eval_batch_size 16

### SIND dataset ###

# create a directory to save trained models
mkdir ../trained_models/sind_bert

# To train a B-TSort model
python model.py --data_dir ../sind_data/ --output_dir ../trained_models/sind_bert/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 16

# To run inference on the trained model use
# This script creates a test_results.tsv file and 
# saves it in the data_sir path provided. 
# This is later useful to get the sentence orders 
# for the document using topological sort script.
python model.py --data_dir ../sind_data/ --output_dir ../
trained_models/sind_bert/ --do_test --per_gpu_eval_batch_size 16

# If you have trained your own model then use the desirable checkpoint
python model.py --data_dir ../sind_data/ --output_dir ../trained_models/sind_bert/checkpoint-X/ --do_test --per_gpu_eval_batch_size 16