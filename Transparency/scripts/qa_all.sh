output_path=./outputs/qa
#qqp snli babi_1 babi_2 babi_3
gpu=0
for dataset_name in qqp snli babi_1
do
  for attention_type in tanh dot
  do
    for encoder_type in cnn lstm
    do
      python train_and_run_experiments_qa.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --attention ${attention_type} --encoder ${encoder_type} --gpu $gpu;
    done
  done
done