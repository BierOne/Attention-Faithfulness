# All exp methods: ours_no_lrp transformer_att partial_lrp raw_attn rollout attn_grad attn_norm inputGrad ig rand
# Metrics: Violation AUCTP Comprehensiveness RC Sufficiency

b_size=128
samples=10000
mask_size=2
gpu=1
workers=4
for method_name in ours_no_lrp 
do
  for exp_metric in Violation
  do
    for delta_type in zeros_mask slice_out att_mask
    do
      CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=`pwd` python lxmert/lxmert/perturbation.py  \
      --COCO_path /home/lyb/vqa_data/mscoco/val2014/ \
      --method $method_name \
      --numWorkers $workers \
      --is-text-pert False \
      --is-positive-pert True \
      --num-samples=$samples \
      --load_raw_img False \
      --load_cached_exp=True \
      --only_perturbation=True \
      --delta_type=$delta_type \
      --exp_metric=$exp_metric \
      --mask_size=$mask_size \
      --batchSize=$b_size
    done
  done
done