# All exp methods: ours_no_lrp ig inputGrad transformer_attribution partial_lrp raw_attn attn_grad rollout attn_norm rand0 rand1 rand2

samples=10000
for text in False
do
  for method_name in rollout inputGrad
  do
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=`pwd` python VisualBERT/run.py \
    --method=$method_name \
    --is-text-pert=$text \
    --is-positive-pert=True \
    --load_cached_exp=False \
    --num-samples=$samples \
    config=projects/visual_bert/configs/vqa2/defaults.yaml \
    model=visual_bert \
    dataset=vqa2 \
    run_type=val \
    checkpoint.resume_zoo=visual_bert.finetuned.vqa2.from_coco_train \
    env.data_dir=./env/data_dir \
    training.num_workers=3 \
    training.batch_size=1 \
    training.trainer=mmf_pert \
    training.seed=1234
  done
done