from lxmert.lxmert.src.tasks import vqa_data
from lxmert.lxmert.src.modeling_frcnn import GeneralizedRCNN
import lxmert.lxmert.src.vqa_utils as utils
from lxmert.lxmert.src.processing_image import Preprocess
from transformers import LxmertTokenizer
from lxmert.lxmert.src.huggingface_lxmert import LxmertForQuestionAnswering
from lxmert.lxmert.src.lxmert_lrp import LxmertForQuestionAnswering as LxmertForQuestionAnsweringLRP
from tqdm import tqdm
from lxmert.lxmert.src.ExplanationGenerator import GeneratorOurs, GeneratorBaselines, GeneratorOursAblationNoAggregation
import random
from lxmert.lxmert.src.param import args

import os, pickle
import h5py
from os.path import exists
from torch import cat, nn
import torch, json
import numpy as np

OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

class ModelPert:
    def __init__(self, COCO_val_path, use_lrp=False, load_raw_img=False):
        self.COCO_VAL_PATH = COCO_val_path
        self.frcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn_cfg.MODEL.DEVICE = "cuda"
        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg)
        self.image_preprocess = Preprocess(self.frcnn_cfg)



    def _create_img_id_to_idx(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(self.image_features_path, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        return coco_id_to_index



    def extract_features(self, item):
        # run frcnn
        image_file_path = self.COCO_VAL_PATH + item['img_id'] + '.jpg'
        self.image_file_path = image_file_path
        images, sizes, scales_yx = self.image_preprocess(image_file_path)
        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections= self.frcnn_cfg.max_detections,
            return_tensors="pt"
        )
        return output_dict.get("normalized_boxes"), output_dict.get("roi_features"), output_dict.get("boxes"), \
               output_dict.get("sizes"), item['coco_id']


def main(args):
    model_pert = ModelPert(args.COCO_path, use_lrp=True, load_raw_img=args.load_raw_img)
    vqa_dataset = vqa_data.VQADataset(splits="valid")
    method_name = args.method


    items = vqa_dataset.data
    random.seed(1234)
    r = list(range(len(items)))
    random.shuffle(r)

    if args.num_samples < 0:
        pert_samples_indices = r
    else:
        pert_samples_indices = r[:args.num_samples]
    iterator = tqdm([vqa_dataset.data[i] for i in pert_samples_indices])

    dir_path = "path_to_image_featues/lxmert/{}".format(args.task)
    file_name = "val_36.h5"
    h5_path = os.path.join(dir_path, file_name)
    features_shape = (len(pert_samples_indices), 36, 2048)
    boxes_shape = (len(pert_samples_indices), 36, 4)

    img_id_to_idx = {}
    with h5py.File(h5_path, mode='w', libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float32')
        normized_boxes = fd.create_dataset('normized_boxes', shape=boxes_shape, dtype='float32')
        boxes = fd.create_dataset('boxes', shape=boxes_shape, dtype='float32')
        sizes = fd.create_dataset('widths', shape=(features_shape[0],2), dtype='int32')

        if args.task=='vqa':
            coco_ids = fd.create_dataset('ids', shape=(features_shape[0],), dtype='int32')

        for index, item in enumerate(iterator):
            normized_box, img_features, box, size, coco_id = model_pert.extract_features(item)
            normized_boxes[index, :] = normized_box
            features[index, :] = img_features
            boxes[index, :] = box
            if args.task == 'vqa':
                coco_ids[index] = coco_id
            else:
                img_id_to_idx[coco_id] = index
            sizes[index, :] = size

    with open(os.path.join(dir_path, 'img_id2idx.json'), 'w') as f:
        json.dump(img_id_to_idx, f)

if __name__ == "__main__":
    main(args)