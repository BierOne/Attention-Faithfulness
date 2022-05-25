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
import torch
import numpy as np

OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"



def get_mask_ratios(exp_metric):
    use_ratio = True
    if exp_metric == "AUCTP": # include completeness
        mask_ratios = np.linspace(0, 90, num=10)
        mask_ratios[0] = 5 # array([0.05, 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9])
        mask_ratios = mask_ratios / 100
    elif exp_metric == "Sufficiency": # include completeness
        mask_ratios = np.array([0.5, 0.8, 0.9, 0.95]) # increasing order
    elif exp_metric == "Comprehensiveness": # include completeness
        mask_ratios = np.array([0.05, 0.1, 0.2, 0.5]) # decending order
    else:
        mask_ratios = 0
        use_ratio = False

    return mask_ratios, use_ratio

def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros_like(labels).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores.sum(dim=1)



class ModelPert:
    def __init__(self, args, use_lrp=False, load_raw_img=False, split="valid"):
        self.vqa_dataset = vqa_data.get_vqa_loader(args, split=split)
        self.vqa_answers = self.vqa_dataset.dataset.label2ans

        if use_lrp:
            self.lxmert_vqa = LxmertForQuestionAnsweringLRP.from_pretrained("unc-nlp/lxmert-{}-uncased".format(args.task)).to("cuda")
        else:
            self.lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-{}-uncased".format(args.task)).to("cuda")
        print("unc-nlp/lxmert-{}-uncased".format(args.task))
        self.lxmert_vqa.eval()
        self.model = self.lxmert_vqa

        self.mask_ratios, self.use_ratio = 0, False
        self.features = None
        self.image_boxes_len = 36
        self.sigmoid, self.softmax = nn.Sigmoid(), nn.Softmax(dim=1),

        self.mask_size = args.mask_size
        self.b_size = args.batch_size
        self.num_samples = args.num_samples
        print("num_samples: ", args.num_samples)




    def perturbation_image(self, cached_exp_scores, exp_metric="AUCTP", delta_type="zeros_mask"):
        mask_ratios, use_ratio = get_mask_ratios(exp_metric)
        self.mask_ratios, self.use_ratio = mask_ratios, use_ratio
        mask_size = self.mask_size
        if exp_metric == 'Violation':
            mask_size = 1
            print('Violation mask size must be 1')

        if use_ratio:
            # in case of quite long sequence, we use num_slots to reduce iterations
            num_slots = len(mask_ratios)
        else:
            num_slots = (self.image_boxes_len // mask_size)

        if num_slots>20:
            num_slots = 20

        idxes, answ, q_ids, correctness = [torch.tensor((), dtype=torch.int32) for _ in range(4)]
        accs, spccs, weights, nongt_weights, preds = [torch.tensor((), dtype=torch.float32) for _ in
                                                                       range(5)]
        top_ans_deltas, newpred_accs, all_pred_deltas = [torch.tensor((), dtype=torch.float32) for _ in range(3)]
        for b_idx, (index, features, spatials, question, q_len, answer, question_id, att_mask, type_ids) in \
                enumerate(tqdm(self.vqa_dataset, total=self.num_samples//self.b_size)):
            b_end = (b_idx+1)*self.b_size
            if b_end > self.num_samples:
                break
            exp_scores = cached_exp_scores[(b_idx * self.b_size):b_end]
            exp_scores = exp_scores.cuda()
            # print(exp_scores.shape)

            features = features.cuda(non_blocking=True)
            spatials = spatials.cuda(non_blocking=True)
            question = question.cuda(non_blocking=True)
            answer = answer.cuda(non_blocking=True)
            att_mask = att_mask.cuda(non_blocking=True)
            type_ids = type_ids.cuda(non_blocking=True)

            image_features = features.clone().detach()
            image_bboxes = spatials.clone().detach()
            output = self.lxmert_vqa(
                input_ids=question,
                attention_mask=att_mask,
                visual_feats=features,
                visual_pos=spatials,
                token_type_ids=type_ids,
                return_dict=True,
                output_attentions=False,
            )
            pred = output.question_answering_score.data
            acc = compute_score_with_logits(pred, answer)  # b
            # print(acc.mean())
            probs = self.softmax(pred)
            topAns_prob, topAns_pred_ind = probs.topk(k=1, dim=1)

            # find top step boxess
            if exp_metric == "Sufficiency":  # increasing order
                a1_max, a1_max_ind = exp_scores.neg().topk(k=self.image_boxes_len, dim=-1)
                a1_max = a1_max.neg()
            elif exp_metric == "RC":
                a1_max, a1_max_ind = exp_scores.abs().topk(k=self.image_boxes_len, dim=-1)
            elif exp_metric == "Violation":
                _, a1_max_ind = exp_scores.abs().topk(k=self.image_boxes_len, dim=-1)
                a1_max = exp_scores.gather(dim=1, index=a1_max_ind) # remain the sign
            else:
                a1_max, a1_max_ind = exp_scores.topk(k=self.image_boxes_len, dim=-1)

            batch_topAns_deltas, batch_newpred_acc, batch_pred_deltas, batch_weights = \
                    [torch.zeros((self.b_size, num_slots)) for _ in range(4)]

            for step_idx, rank in enumerate(range(num_slots)):
                if use_ratio:
                    ratio = mask_ratios[rank]
                    mask_size = round(ratio * self.image_boxes_len)  # [b]
                    exp_rank_ind = a1_max_ind[:, :mask_size]  # e.g., top-[5%, 10%, 50%] or last-[95%, 80%]
                    exp_rank_weights = a1_max[:, :mask_size]
                else:
                    exp_rank_ind = a1_max_ind[:, rank * mask_size:mask_size * (rank + 1)]
                    exp_rank_weights = a1_max[:, rank * mask_size:mask_size * (rank + 1)]

                batch_weights[:, rank] = exp_rank_weights.sum(dim=-1).cpu().data

                masks = torch.ones_like(exp_scores)
                masks = masks.scatter(dim=1, index=exp_rank_ind, value=0).long()  # [b, o]
                # print(batch['image_info_0']['max_features'], batch['image_info_0']['num_boxes'])
                visual_attention_mask = None
                if delta_type == "zeros_mask": # corresponding to MASK_Token
                    mask_features = image_features * (masks.unsqueeze(2))
                    mask_spatials = image_bboxes * (masks.unsqueeze(2))

                elif delta_type == "slice_out":
                    # remove the top step boxes from the batch info
                    mask_features = image_features[masks.bool()].view(-1,self.image_boxes_len-mask_size,2048)
                    mask_spatials = image_bboxes[masks.bool()].view(-1,self.image_boxes_len-mask_size,4)

                elif delta_type == "att_mask":
                    mask_features = image_features
                    mask_spatials = image_bboxes
                    visual_attention_mask = masks

                new_output = self.lxmert_vqa(
                    input_ids=question,
                    attention_mask=att_mask,
                    visual_feats=mask_features,
                    visual_pos=mask_spatials,
                    token_type_ids=type_ids,
                    visual_attention_mask=visual_attention_mask,
                    return_dict=True,
                    output_attentions=False,
                )
                new_pred = new_output.question_answering_score.data
                new_probs = self.softmax(new_pred)

                new_acc = compute_score_with_logits(new_pred, answer)  # b
                batch_newpred_acc[:, rank] = new_acc.cpu()
                # print(new_acc.mean())

                new_topAns_prob = new_probs.gather(dim=1, index=topAns_pred_ind)  # [b, k]
                batch_topAns_deltas[:, rank] = (topAns_prob - new_topAns_prob).mean(dim=1).detach().cpu()  # b
                batch_pred_deltas[:, rank] = ((probs) - (new_probs)).abs().mean(dim=1).detach().cpu()  # b

            top_ans_deltas = torch.cat([top_ans_deltas, batch_topAns_deltas.detach().cpu()])
            newpred_accs = torch.cat([newpred_accs, batch_newpred_acc.detach().cpu()])
                 = torch.cat([all_pred_deltas, batch_pred_deltas.detach().cpu()])
            preds = torch.cat([preds, topAns_prob.detach().cpu()])
            weights = torch.cat([weights, batch_weights.detach().cpu()])
            accs = torch.cat([accs, acc.view(-1).detach().cpu()])
            q_ids = torch.cat([q_ids, question_id.view(-1)])

        print("mean prob:", preds.mean())
        return top_ans_deltas, newpred_accs, all_pred_deltas, preds, weights, accs, q_ids



def main(args):
    input_grad = True if args.method in ["inputGrad", "ig"] else False
    model_pert = ModelPert(args, use_lrp=True)

    # ours = GeneratorOurs(model_pert)
    # baselines = GeneratorBaselines(model_pert)

    method = args.method
    load_cached_exp = args.load_cached_exp
    exp_metrics = args.exp_metric
    delta_types = args.delta_type

    if exp_metrics == "all":
        exp_metrics = ["AUCTP", "Sufficiency", "Violation"]
    else:
        exp_metrics = [exp_metrics]

    if delta_types == "all":
        delta_types = ["zeros_mask", "slice_out", "att_mask"]
    else:
        delta_types = [delta_types]

    only_perturbation = args.only_perturbation

    print("test type {0} expl type {1}".format("image", method))

    has_cached_exp, text_cached_exp_scores, img_cached_exp_scores = False, [], []
    exp_path = "./data/lxmert_{}2".format(args.task)
    print(exp_path)
    create_dir(exp_path)

    img_cached_exp_path = os.path.join(exp_path, method + "_image.pkl")
    text_cached_exp_path = os.path.join(exp_path, method + "_text.pkl")

    if load_cached_exp and exists(img_cached_exp_path):
        print("load {} exp scores from {}".format(method, img_cached_exp_path))
        cached_data = pickle.load(open(img_cached_exp_path, 'rb'))
        if 'rand' not in method:
            q_ids = cached_data['q_ids']
            q_ids = np.array(q_ids, dtype=np.int32)
            assert model_pert.vqa_dataset.dataset.assert_qids(q_ids)
            img_cached_exp_scores = cached_data['img_cached_exp_scores']
        else:
            img_cached_exp_scores = cached_data
        img_cached_exp_scores = torch.from_numpy(img_cached_exp_scores).view(-1, 36)
        print(img_cached_exp_scores.shape)
        has_cached_exp = True
    else:
        print("img_cached_exp_path wrong path")


    save_scores_path = os.path.join(exp_path, method)
    create_dir(save_scores_path)

    for exp_metric in exp_metrics:
        for delta_type in delta_types:
            print("exp_metric {0} delta_type {1}".format(exp_metric, delta_type))
            top_ans_deltas, newpred_accs, all_pred_deltas, preds, weights, accs, q_ids = \
                model_pert.perturbation_image(img_cached_exp_scores, exp_metric=exp_metric, delta_type=delta_type)

            with open(os.path.join(save_scores_path, exp_metric + delta_type +'.pkl'), 'wb') as f:
                print("save {} perturbation scores to {}".format(exp_metric + delta_type, save_scores_path))
                pickle.dump({
                    "top_ans_deltas": top_ans_deltas.numpy(),
                    "newpred_accs": newpred_accs.numpy(),
                    "all_pred_deltas": all_pred_deltas.numpy(),
                    "preds": preds.numpy(),
                    "weights": weights.numpy(),
                    "accs": accs.numpy(),
                    "mask_ratios": model_pert.mask_ratios,
                    "q_ids": q_ids.numpy()
                }, f)
            print(accs.mean())
            print(newpred_accs.mean(dim=0))

if __name__ == "__main__":
    main(args)