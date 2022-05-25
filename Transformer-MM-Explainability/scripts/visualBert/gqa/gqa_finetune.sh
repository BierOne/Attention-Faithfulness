# Since VisualBert-GQA is not provided now, we need first finetune the pretrained model

gpu=2,3
task=gqa
CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=`pwd` python VisualBERT/run.py \
--task=$task \
config=projects/visual_bert/configs/gqa/defaults.yaml \
model=visual_bert \
dataset=gqa \
run_type=train_val \
checkpoint.resume=True \
env.data_dir=./env/data_dir \
training.num_workers=2 \
training.batch_size=8 \
training.trainer=mmf \
training.seed=1234
