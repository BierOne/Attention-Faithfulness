# All exp methods: ours_no_lrp ig inputGrad transformer_attribution partial_lrp raw_attn attn_grad rollout attn_norm rand0 rand1 rand2
# Metrics: Violation AUCTP Comprehensiveness RC Sufficiency

# We can use larger batchsize here since we do not extract explanations again
samples=10000
mask_size=5
b_size=4
task=gqa
for method_name in rollout 
do
  for exp_metric in Violation RC AUCTP Sufficiency
  do
    for delta_type in zeros_mask slice_out att_mask
    do
      CUDA_VISIBLE_DEVICES=1 PYTHONPATH=`pwd` python VisualBERT/run.py --method=$method_name \
      --is-text-pert=False \
      --is-positive-pert=True \
      --load_cached_exp=True \
      --only_perturbation=True \
      --num-samples=$samples \
      --delta_type=$delta_type \
      --exp_metric=$exp_metric \
      --mask_size=$mask_size \
      --b_size=$b_size \
      --task=$task \
      config=projects/visual_bert/configs/gqa/defaults.yaml \
      dataset=gqa \
      model=visual_bert \
      run_type=val \
      checkpoint.resume_file=path_to_model/Transformer-MM-Explainability/save/gqa_visual_bert/visual_bert_final.pth \
      env.data_dir=./env/data_dir \
      training.num_workers=3 \
      training.batch_size=$b_size \
      training.trainer=mmf_pert \
      training.seed=1234
    done
  done
done