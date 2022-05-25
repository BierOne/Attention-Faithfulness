# sst yelp 20News_sports AgNews
# tanh dot
# cnn lstm
gpu=0
output_path=./outputs/mc
echo $output_path

for dataset_name in sst yelp 20News_sports AgNews
do
  for attention_type in tanh dot
  do
    for encoder_type in cnn lstm
    do
      python train_and_run_experiments_bc.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --attention ${attention_type} --encoder ${encoder_type} --gpu $gpu;
    done
  done
done