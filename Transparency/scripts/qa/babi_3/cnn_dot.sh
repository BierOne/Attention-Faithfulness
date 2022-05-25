dataset_name=babi_3 # [sst, imdb, 20News_sports, tweet, Anemia, Diabetes, AgNews]
output_path=./outputs
encoder_type=cnn
attention_type=dot
gpu=2
python train_and_run_experiments_qa.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --attention ${attention_type} --encoder ${encoder_type} --gpu ${gpu}