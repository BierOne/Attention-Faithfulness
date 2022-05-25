# All exp methods: ours_no_lrp ig inputGrad transformer_attribution partial_lrp raw_attn attn_grad rollout attn_norm rand0 rand1 rand2

gpu=2
samples=10000
task=gqa
for text in False
do
  for method_name in partial_lrp transformer_attribution
  do
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=`pwd` python VisualBERT/run.py \
    --method=$method_name \
    --is-text-pert=$text \
    --is-positive-pert=True \
    --load_cached_exp=False \
    --task=$task \
    --num-samples=$samples \
    config=projects/visual_bert/configs/gqa/defaults.yaml \
    model=visual_bert \
    dataset=gqa \
    run_type=val \
    checkpoint.resume_file=path_to_model/Transformer-MM-Explainability/save/gqa_visual_bert/visual_bert_final.pth \
    env.data_dir=./env/data_dir \
    training.num_workers=3 \
    training.batch_size=1 training.trainer=mmf_pert training.seed=1234
  done
done