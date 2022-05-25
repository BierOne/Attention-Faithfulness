dataset_name=cnn
output_path=./outputs
encoder_type=lstm
attention_type=dot
gpu=2
python train_and_run_experiments_qa.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --attention ${attention_type} --encoder ${encoder_type}