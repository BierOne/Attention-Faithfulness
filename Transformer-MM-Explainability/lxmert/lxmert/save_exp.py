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
import h5py, json
from os.path import exists
from torch import cat, nn
import torch
import numpy as np

OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"

# the features are probided by the original author: airsplay/lxmert;
# Or you can extract it by yourself using "extract_img_features.py"
# You can also use the option, load_raw_img, in ModelPert (which would be relatively slow).
NEW_IMAGE_FEATURE_PATH = "path_to_image_featues/lxmert/{}/val_36.h5".format(args.task)
print(NEW_IMAGE_FEATURE_PATH)

class ModelPert:
    def __init__(self, args, COCO_val_path, use_lrp=False, load_raw_img=False):
        self.COCO_VAL_PATH = COCO_val_path
        # self.vqa_answers = utils.get_data(VQA_URL)
        self.vqa_answers = json.load(open("data/{}/trainval_label2ans.json".format(args.task)))
        self.image_features_path = NEW_IMAGE_FEATURE_PATH
        self.load_raw_img = load_raw_img

        # load models and model components
        if load_raw_img:
            self.frcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
            self.frcnn_cfg.MODEL.DEVICE = "cuda"
            self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg)
            self.image_preprocess = Preprocess(self.frcnn_cfg)
        else:
            self.features_file = h5py.File(self.image_features_path, 'r', swmr=True)
            if args.task == 'vqa':
                self.img_id2idx = self._create_img_id_to_idx()
            else:
                # "GQA"
                self.img_id2idx = json.load(open(os.path.join("path_to_image_featues/transformer_lxmert",
                                                        args.task, "img_id2idx.json"), 'r'))


        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        if use_lrp:
            self.lxmert_vqa = LxmertForQuestionAnsweringLRP.from_pretrained("unc-nlp/lxmert-{}-uncased".format(args.task)).to("cuda")
        else:
            self.lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-{}-uncased".format(args.task)).to("cuda")
        print("load model from: ", "unc-nlp/lxmert-{}-uncased".format(args.task))

        self.lxmert_vqa.eval()
        self.model = self.lxmert_vqa

        self.pert_steps = [0, 0.05, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        self.pert_acc = [0] * len(self.pert_steps)
        self.features = None
        self.sigmoid, self.softmax = nn.Sigmoid(), nn.Softmax(dim=1),


    def _create_img_id_to_idx(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(self.image_features_path, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        return coco_id_to_index


    def load_image(self, img_idx):
        feature = self.features_file['features'][img_idx]
        # spatials = self.features_file['boxes'][img_idx]
        spatials = self.features_file['normized_boxes'][img_idx]
        return torch.from_numpy(feature), torch.from_numpy(spatials)



    def forward_ig(self, item, folds=10):
        # run frcnn
        if self.load_raw_img:
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
            normalized_boxes = output_dict.get("normalized_boxes")
            self.features = output_dict.get("roi_features")
            self.image_boxes_len = self.features.shape[1]
            self.bboxes = output_dict.get("boxes")
            self.features = self.features.to("cuda")
            # print(self.features.shape, self.bboxes.shape)

        else:
            img_idx = self.img_id2idx[item['coco_id']]
            self.features, normalized_boxes = self.load_image(img_idx)
            self.features = self.features.to("cuda").unsqueeze(0)
            normalized_boxes = normalized_boxes.unsqueeze(0)
            self.image_boxes_len = normalized_boxes.shape[1]


        inputs = self.lxmert_tokenizer(
            item['sent'],
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        self.question_tokens = self.lxmert_tokenizer.convert_ids_to_tokens(inputs.input_ids.flatten())
        self.text_len = len(self.question_tokens)
        # Very important that the boxes are normalized

        # image_features = self.features.clone().detach()
        self.features.requires_grad_()
        grad = None
        for i in range(1, folds+1):
            self.model.zero_grad()
            ig_ratio = i / folds
            output = self.lxmert_vqa(
                input_ids=inputs.input_ids.to("cuda"),
                attention_mask=inputs.attention_mask.to("cuda"),
                visual_feats=self.features * ig_ratio,
                visual_pos=normalized_boxes.to("cuda"),
                token_type_ids=inputs.token_type_ids.to("cuda"),
                return_dict=True,
                output_attentions=False,
            )

            output = output.question_answering_score
            index = np.argmax(output.cpu().data.numpy(), axis=-1)
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0, index] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            if i == 1:
                grad = torch.autograd.grad(one_hot, self.features, create_graph=False, retain_graph=True)[0]
            else:
                grad += torch.autograd.grad(one_hot, self.features, create_graph=False, retain_graph=True)[0]
            # print(input['image_feature_0'].grad[0, :5])


        cam = (self.features.detach()) * (grad.data) # [n, tokens, scores]
        cam = cam.sum(dim=-1) # [1, num_tokens]

        return cam



    def forward(self, item, save_values=False):
        self.image_id = item['img_id']
        # run frcnn
        if self.load_raw_img:
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
            normalized_boxes = output_dict.get("normalized_boxes")
            self.features = output_dict.get("roi_features")
            self.image_boxes_len = self.features.shape[1]
            self.bboxes = output_dict.get("boxes")
            self.features = self.features.to("cuda")
            # print(self.features.shape, self.bboxes.shape)

        else:
            img_idx = self.img_id2idx[item['coco_id']]
            self.features, normalized_boxes = self.load_image(img_idx)
            self.features = self.features.to("cuda").unsqueeze(0)
            normalized_boxes = normalized_boxes.unsqueeze(0)
            self.image_boxes_len = normalized_boxes.shape[1]


        inputs = self.lxmert_tokenizer(
            item['sent'],
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        self.question_tokens = self.lxmert_tokenizer.convert_ids_to_tokens(inputs.input_ids.flatten())
        self.text_len = len(self.question_tokens)
        # Very important that the boxes are normalized

        self.output = self.lxmert_vqa(
            input_ids=inputs.input_ids.to("cuda"),
            attention_mask=inputs.attention_mask.to("cuda"),
            visual_feats=self.features,
            visual_pos=normalized_boxes.to("cuda"),
            token_type_ids=inputs.token_type_ids.to("cuda"),
            return_dict=True,
            output_attentions=False,
            save_values=save_values,
        )
        return self.output


    def perturbation_image(self, item, cam_image, cam_text, is_positive_pert=False):
        if is_positive_pert:
            pos = -1
            cam_image = cam_image * (-1)
        else:
            pos = 1
        # run frcnn
        if self.load_raw_img:
            image_file_path = self.COCO_VAL_PATH + item['img_id'] + '.jpg'
            images, sizes, scales_yx = self.image_preprocess(image_file_path)
            output_dict = self.frcnn(
                images,
                sizes,
                scales_yx=scales_yx,
                padding="max_detections",
                max_detections= self.frcnn_cfg.max_detections,
                return_tensors="pt"
            )
            self.normalized_boxes = output_dict.get("normalized_boxes")
            self.features = output_dict.get("roi_features")
            self.image_boxes_len = self.features.shape[1]
            self.bboxes = output_dict.get("boxes")
            self.features = self.features.to("cuda")
            # print(self.features.shape, self.bboxes.shape)
        else:
            img_idx = self.img_id2idx[item['coco_id']]
            self.features, self.normalized_boxes = self.load_image(img_idx)
            self.features = self.features.to("cuda").unsqueeze(0)
            self.normalized_boxes = self.normalized_boxes.unsqueeze(0)
            self.image_boxes_len = self.normalized_boxes.shape[1]

        inputs = self.lxmert_tokenizer(
            item['sent'],
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        # Very important that the boxes are normalized
        _, bboxes_indices = cam_image.abs().topk(k=self.image_boxes_len, dim=-1)

        is_vio = None
        for step_idx, step in enumerate(self.pert_steps):
            curr_num_tokens = int((1 - step) * self.image_boxes_len)
            top_bboxes_indices = bboxes_indices[:curr_num_tokens].cpu().data.numpy()

            curr_features = self.features[:, top_bboxes_indices, :]
            curr_pos = self.normalized_boxes[:, top_bboxes_indices, :]

            output = self.lxmert_vqa(
                input_ids=inputs.input_ids.to("cuda"),
                attention_mask=inputs.attention_mask.to("cuda"),
                visual_feats=curr_features.to("cuda"),
                visual_pos=curr_pos.to("cuda"),
                token_type_ids=inputs.token_type_ids.to("cuda"),
                return_dict=True,
                output_attentions=False,
                save_values=False,
            )

            if step_idx == 0:
                probs = self.sigmoid(output.question_answering_score)
                topAns_prob, topAns_pred_ind = probs.topk(k=1, dim=1)

            if step_idx == 1:
                new_probs = self.sigmoid(output.question_answering_score)
                new_topAns_prob = new_probs.gather(dim=1, index=topAns_pred_ind)  # [b, k]
                batch_topAns_deltas = (topAns_prob - new_topAns_prob).mean(dim=1).detach().cpu()  # b
                batch_weights = pos * cam_image.gather(dim=-1, index=bboxes_indices[curr_num_tokens:])
                is_vio = ((batch_topAns_deltas.sign()) * (batch_weights.sum().sign().cpu())) < 0
                # print(batch_topAns_deltas, new_topAns_prob, is_vio, batch_weights, bbox_scores.max().item()*pos)
                break

            answer = self.vqa_answers[output.question_answering_score.argmax()]
            accuracy = item["label"].get(answer, 0)
            self.pert_acc[step_idx] += accuracy

        return self.pert_acc, is_vio.sum().item()


def main(args):
    input_grad = True if args.method in ["inputGrad", "ig"] else False
    model_pert = ModelPert(args, args.COCO_path, use_lrp=True, load_raw_img=args.load_raw_img)
    ours = GeneratorOurs(model_pert)
    baselines = GeneratorBaselines(model_pert)
    # oursNoAggAblation = GeneratorOursAblationNoAggregation(model_pert)
    vqa_dataset = vqa_data.VQADataset(splits="valid")
    method_name = args.method
    load_cached_exp = args.load_cached_exp


    items = vqa_dataset.data
    random.seed(1234)
    r = list(range(len(items)))
    random.shuffle(r)
    # print(r[:100])

    pert_samples_indices = r[:args.num_samples]
    iterator = tqdm([vqa_dataset.data[i] for i in pert_samples_indices])

    test_type = "positive" if args.is_positive_pert else "negative"
    modality = "text" if args.is_text_pert else "image"
    print("runnig {0} pert test for {1} modality with method {2}".format(test_type, modality, args.method))


    has_cached_exp, text_cached_exp_scores, img_cached_exp_scores = False, [], []
    exp_path = "./data/lxmert_{}2".format(args.task)
    print(exp_path)
    img_cached_exp_path = os.path.join(exp_path, method_name + "_image.pkl")
    text_cached_exp_path = os.path.join(exp_path, method_name + "_text.pkl")

    if load_cached_exp and exists(img_cached_exp_path):
        print("load {} exp scores from {}".format(method_name, img_cached_exp_path))
        img_cached_exp_scores = pickle.load(open(img_cached_exp_path, 'rb'))
        img_cached_exp_scores = torch.from_numpy(img_cached_exp_scores)
        has_cached_exp = True

    violators = 0

    q_ids = []
    for index, item in enumerate(iterator):
        q_ids.append(item['question_id'])
        if not has_cached_exp:
            if method_name == 'transformer_att':
                R_t_t, R_t_i = baselines.generate_transformer_attr(item)
            elif method_name == 'attn_gradcam':
                R_t_t, R_t_i = baselines.generate_attn_gradcam(item)
            elif method_name == 'partial_lrp':
                R_t_t, R_t_i = baselines.generate_partial_lrp(item)
            elif method_name == 'raw_attn':
                R_t_t, R_t_i = baselines.generate_raw_attn(item)
            elif method_name == 'rollout':
                R_t_t, R_t_i = baselines.generate_rollout(item)

            elif method_name == "ours_with_lrp_no_normalization":
                R_t_t, R_t_i = ours.generate_ours(item, normalize_self_attention=False)
            elif method_name == "ours_no_lrp":
                R_t_t, R_t_i = ours.generate_ours(item, use_lrp=False)
            elif method_name == "ours_no_lrp_no_norm":
                R_t_t, R_t_i = ours.generate_ours(item, use_lrp=False, normalize_self_attention=False)
            elif method_name == "ours_with_lrp":
                R_t_t, R_t_i = ours.generate_ours(item, use_lrp=True)

            elif method_name == "attn_norm":
                R_t_t, R_t_i = baselines.generate_att_norm(item)
            elif method_name == "ig":
                R_t_t, R_t_i = baselines.generate_ig(item)
            elif method_name == "inputGrad":
                R_t_t, R_t_i = baselines.generate_input_grad(item)
            elif method_name == "attn_grad":
                R_t_t, R_t_i = baselines.generate_attn_grad(item)
            else:
                print("Please enter a valid method name")
                return
            img_cached_exp_scores.append(R_t_i[0].detach().cpu())
            text_cached_exp_scores.append(R_t_t[0].detach().cpu().numpy())
        else:
            R_t_i = img_cached_exp_scores[index]
            R_t_t = None

        cam_image = R_t_i[0]
        
        curr_pert_result, is_vio = model_pert.perturbation_image(item, cam_image, None, args.is_positive_pert)
        violators += is_vio
        curr_pert_result = [round(res / (index+1) * 100, 2) for res in curr_pert_result]
        iterator.set_description("Acc: {}".format(curr_pert_result))
        torch.cuda.empty_cache()

    print("violatio:{}".format(violators/args.num_samples))

    if not has_cached_exp:
        with open(img_cached_exp_path, 'wb') as f:
            img_cached_exp_scores = torch.stack(img_cached_exp_scores, dim=0).numpy()
            print(img_cached_exp_scores.shape)
            print("save {} exp scores to {}".format(method_name, img_cached_exp_path))
            pickle.dump({
                "img_cached_exp_scores" : img_cached_exp_scores,
                "q_ids": q_ids
            }, f)

if __name__ == "__main__":
    main(args)