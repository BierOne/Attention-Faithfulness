samples=-1
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=`pwd` python lxmert/lxmert/extract_img_features.py  \
--COCO_path /home/lyb/vqa_data/gqa/images/  \
--num-samples=$samples \
--task gqa
