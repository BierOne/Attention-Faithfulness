dataset_name=yelp # ['imdb', 'sst', 'yelp', '20News_sports', 'amazon']
output_path=./outputs
encoder_type=cnn
attention_type=dot
gpu=2
python train_and_run_experiments_bc.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --attention ${attention_type} --encoder ${encoder_type} --gpu ${gpu}