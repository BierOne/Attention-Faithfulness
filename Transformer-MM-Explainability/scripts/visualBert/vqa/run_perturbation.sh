# All exp methods: ours_no_lrp ig inputGrad transformer_attribution partial_lrp raw_attn attn_grad rollout attn_norm rand0 rand1 rand2
# Metrics: Violation AUCTP Comprehensiveness RC Sufficiency

# We can use larger batchsize here since we do not extract explanations again
samples=10000
mask_size=5
b_size=5
for method_name in ours_no_lrp ig inputGrad transformer_attribution partial_lrp raw_attn
do
  for exp_metric in Violation
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
      config=projects/visual_bert/configs/vqa2/defaults.yaml \
      dataset=vqa2 \
      model=visual_bert \
      run_type=val \
      checkpoint.resume_zoo=visual_bert.finetuned.vqa2.from_coco_train \
      env.data_dir=./env/data_dir \
      training.num_workers=3 \
      training.batch_size=$b_size \
      training.trainer=mmf_pert \
      training.seed=1234
    done
  done
done