
# All exp methods: ours_no_lrp transformer_att partial_lrp raw_attn rollout attn_grad attn_norm inputGrad ig rand

samples=10000
for text in False
do
  for method_name in ours_no_lrp transformer_att partial_lrp raw_attn rollout attn_grad attn_norm inputGrad ig
  do
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=`pwd` python lxmert/lxmert/save_exp.py  \
    --COCO_path /home/lyb/vqa_data/mscoco/val2014/ \
    --method $method_name \
    --is-text-pert $text \
    --is-positive-pert True \
    --load_cached_exp=False \
    --num-samples=$samples \
    --load_raw_img False
  done
done