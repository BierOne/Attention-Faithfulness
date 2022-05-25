# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
import random, h5py
from transformers import LxmertTokenizer
import lxmert.lxmert.src.vqa_utils as utils

from ..param import args
from ..utils import load_obj_tsv

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"

# The path to data and image features.
VQA_DATA_ROOT = 'data/vqa/'

DIR_FEATURE_PATH = "path_to_img_features/lxmert"

MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'
SPLIT2NAME = {
    'train': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'test2015',
}


class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        self.task = args.task
        for split in self.splits:
            self.data.extend(json.load(open("data/{}/{}.json".format(args.task,split))))
        if self.task == 'vqa':
            for i, item in enumerate(self.data):
                item["coco_id"] = int(item["img_id"].split('_')[-1])
        else:
            for i, item in enumerate(self.data):
                item["coco_id"] = item["img_id"]
        print(self.data[:10])


        print("Load %d data from split(s) %s. (%s)" % (len(self.data), self.name, args.task))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("data/{}/trainval_ans2label.json".format(args.task)))
        self.label2ans = json.load(open("data/{}/trainval_label2ans.json".format(args.task)))
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)



from torch.utils.data import DataLoader
def get_vqa_loader(args, split):
    """ Returns a data loader for the desired split """
    dataset = Perturabtion_VQADataset(split)
    print(args.num_workers)
    # We cannot load image features with batch_size due to the restrictions in h5py (increasing and no duplicate indexes)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # only shuffle the data in training
        pin_memory=True,
        num_workers=int(args.num_workers),
    )
    return loader



class Perturabtion_VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')
        self.task = args.task
        # Answers
        # self.label2ans = utils.get_data(VQA_URL)
        # self.ans2label = {a:idx for idx, a in enumerate(self.label2ans)}
        print("data/{}/trainval_label2ans.json".format(args.task))
        self.ans2label = json.load(open("data/{}/trainval_ans2label.json".format(args.task)))
        self.label2ans = json.load(open("data/{}/trainval_label2ans.json".format(args.task)))
        self.num_candidate_answers = len(self.label2ans)

        # Loading datasets
        all_data = []
        for split in self.splits:
            all_data.extend(json.load(open("data/{}/{}.json".format(args.task,split))))

        # Convert list to dict (for evaluation)
        self.id2datum = {}
        self.questions, self.q_lens, self.answers, self.qids, self.img_ids, \
        self.att_masks, self.type_ids = [[] for _ in range(7)]

        random.seed(1234)
        r = list(range(len(all_data)))
        random.shuffle(r)

        pert_samples_indices = r[:args.num_samples]
        self.data = [all_data[i] for i in pert_samples_indices]

        print("Load %d data from split(s) %s. (%s)" % (len(self.data), self.name, args.task))
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

        for datum in self.data:
            qid = datum['question_id']
            if self.task == 'vqa':
                coco_id = int(datum["img_id"].split('_')[-1])
            else:
                coco_id = datum["img_id"]
            self.id2datum[qid] = datum
            question, mask, type_id, q_len = self.lxmert_encode_question(datum['sent'])
            target = self.encode_label(datum['label'])

            self.img_ids.append(coco_id)
            self.qids.append(qid)
            self.questions.append(question)
            self.att_masks.append(mask)
            self.type_ids.append(type_id)
            self.q_lens.append(q_len)
            self.answers.append(target)


        self.image_features_path = os.path.join(DIR_FEATURE_PATH, args.task, "val_36.h5")
        if self.task == 'vqa':
            self.img_id2idx = self._create_img_id_to_idx()
        else:
            self.img_id2idx = json.load(open(os.path.join(DIR_FEATURE_PATH, args.task, "img_id2idx.json"), 'r'))

        self.features_file = h5py.File(self.image_features_path, 'r', swmr=True)
        self.convert_array()

    def encode_label(self, raw_label):
        target = torch.zeros(self.num_candidate_answers)
        for ans, score in raw_label.items():
            target[self.ans2label[ans]] = score
        return target

    def _create_img_id_to_idx(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(self.image_features_path, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        return coco_id_to_index


    def lxmert_encode_question(self, question):
        encoding = self.lxmert_tokenizer(
            question,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        question_tokens = self.lxmert_tokenizer.convert_ids_to_tokens(encoding["input_ids"].flatten())
        text_len = len(question_tokens)
        return encoding["input_ids"].squeeze(), encoding["attention_mask"], encoding["token_type_ids"], text_len

    def convert_array(self):
        """ referring to https://github.com/pytorch/pytorch/issues/13246,
            each object in list or dict has a refcount, therefore every item would have a refcount.
            This can lead the CPU memory leak in the training (especially multiple workers).
            So conversion of Numpy arrays or tensors are necessary, which can store data as continuous blocks"""
        self.questions = torch.stack(self.questions, dim=0).numpy()
        self.q_lens = torch.LongTensor(self.q_lens).numpy()
        self.answers = torch.stack(self.answers, dim=0).numpy()
        self.qids = np.array(self.qids, dtype=np.int32)
        # self.img_ids = np.array(self.img_ids, dtype=np.int32)
        self.type_ids = torch.cat(self.type_ids, dim=0).numpy()
        self.att_masks = torch.cat(self.att_masks, dim=0).numpy()
        print("Load %d data from split(s) %s." % (len(self.data), self.name))
        print(self.questions.shape, self.qids.shape, self.answers.shape)

    def assert_qids(self, cached_qids):
        # print(self.qids[:10], cached_qids[:10])
        return (self.qids==cached_qids[:self.qids.shape[0]]).all()

    def load_image(self, img_idx):
        feature = self.features_file['features'][img_idx]
        # spatials = self.features_file['boxes'][img_idx]
        spatials = self.features_file['normized_boxes'][img_idx]
        return torch.from_numpy(feature), torch.from_numpy(spatials)


    def __getitem__(self, index):
        img_idx = self.img_id2idx[self.img_ids[index]]
        features, spatials = self.load_image(img_idx)
        question = self.questions[index]
        q_len = self.q_lens[index]
        question_id = self.qids[index]
        answer = self.answers[index]
        att_mask = self.att_masks[index]
        type_id = self.type_ids[index]
        return index, features, spatials, question, q_len, answer, question_id, att_mask, type_id



    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)



"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading detection features to img_data
        img_data = []
        for split in dataset.splits:
            # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
            # It is saved as the top 5K features in val2014_***.tsv
            load_topk = 5000 if (split == 'minival' and topk is None) else topk
            img_data.extend(load_obj_tsv(
                os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
                topk=load_topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


