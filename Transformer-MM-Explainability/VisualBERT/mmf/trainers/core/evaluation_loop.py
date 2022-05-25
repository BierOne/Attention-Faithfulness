# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from abc import ABC
from typing import Any, Dict, Tuple, Type

import os, pickle
from os.path import exists
from torch import cat
import numpy as np
import torch

import torch
from torch import nn
import tqdm
from VisualBERT.mmf.common.meter import Meter
from VisualBERT.mmf.common.report import Report
from VisualBERT.mmf.common.sample import to_device
from VisualBERT.mmf.utils.distributed import is_master
from VisualBERT.mmf.models.transformers.backends import ExplanationGenerator
from VisualBERT import perturbation_arguments

logger = logging.getLogger(__name__)

class TrainerEvaluationLoopMixin(ABC):
    def evaluation_loop(
        self, loader, use_tqdm: bool = False, single_batch: bool = False
    ) -> Tuple[Dict[str, Any], Type[Meter]]:
        meter = Meter()

        with torch.no_grad():
            self.model.eval()
            disable_tqdm = not use_tqdm or not is_master()
            combined_report = None

            for batch in tqdm.tqdm(loader, disable=disable_tqdm):
                report = self._forward(batch)
                self.update_meter(report, meter)

                # accumulate necessary params for metric calculation
                if combined_report is None:
                    combined_report = report
                else:
                    combined_report.accumulate_tensor_fields(
                        report, self.metrics.required_params
                    )
                    combined_report.batch_size += report.batch_size

                if single_batch is True:
                    break

            combined_report.metrics = self.metrics(combined_report, combined_report)
            self.update_meter(combined_report, meter, eval_mode=True)

            # enable train mode again
            self.model.train()

        return combined_report, meter

    def prediction_loop(self, dataset_type: str) -> None:
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        with torch.no_grad():
            self.model.eval()
            logger.info(f"Starting {dataset_type} inference predictions")

            while reporter.next_dataset():
                dataloader = reporter.get_dataloader()

                for batch in tqdm.tqdm(dataloader):
                    prepared_batch = reporter.prepare_batch(batch)
                    prepared_batch = to_device(prepared_batch, torch.device("cuda"))
                    with torch.cuda.amp.autocast(enabled=self.training_config.fp16):
                        model_output = self.model(prepared_batch)
                    report = Report(prepared_batch, model_output)
                    reporter.add_to_report(report, self.model)

            logger.info("Finished predicting")
            self.model.train()


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


class TrainerEvaluationLoopMixinPert(ABC):
    def evaluation_loop(self, loader, on_test_end, use_tqdm: bool = False):
        self.model.eval()
        expl = ExplanationGenerator.SelfAttentionGenerator(self.model)

        sigmoid, softmax = nn.Sigmoid(), nn.Softmax(dim=1),
        method = perturbation_arguments.args.method
        load_cached_exp = perturbation_arguments.args.load_cached_exp
        pert_type = "pos" if perturbation_arguments.args.is_positive_pert else "neg"
        modality = "text" if perturbation_arguments.args.is_text_pert else "image"
        num_samples = perturbation_arguments.args.num_samples
        task = perturbation_arguments.args.task

        exp_metric = perturbation_arguments.args.exp_metric
        delta_type = perturbation_arguments.args.delta_type
        only_v_grad = True if perturbation_arguments.args.method in ["inputGrad", "ig"] else False
        only_perturbation = perturbation_arguments.args.only_perturbation

        method_expl = {"transformer_attribution": expl.generate_transformer_att,
                       "ours_no_lrp": expl.generate_ours,
                       "partial_lrp": expl.generate_partial_lrp,
                       "raw_attn": expl.generate_raw_attn,
                       "rollout": expl.generate_rollout,
                       "attn_grad": expl.generate_attn_grad,
                       "attn_gradcam": expl.generate_attn_gradcam,
                       "attn_norm": expl.generate_attn_norm,
                       "inputGrad": expl.generate_inputGrad,
                       "ig": expl.generate_ig,
                       
                       }

        image_boxes_len, features_dim =  100, 2048  # [b, 100, 2048], [100,4]
        has_cached_exp, cached_exp_scores = False, []
        exp_path = "./data/visual_bert_{}2".format(task)
        cached_exp_path = os.path.join(exp_path, method+"_{}.pkl".format(modality))
        print(cached_exp_path)

        if exists(cached_exp_path) and load_cached_exp:
            print("load {} exp scores from {}".format(method, cached_exp_path))
            cached_exp_scores = pickle.load(open(cached_exp_path, 'rb'))
            # we only keep exp scores for image data
            cached_exp_scores = [e[:, -image_boxes_len:] for e in cached_exp_scores]
            cached_exp_scores = torch.from_numpy(np.vstack(cached_exp_scores))
            has_cached_exp = True


        # saving cams per method for all the samples
        self.model.eval()
        disable_tqdm = not use_tqdm or not is_master()

        mask_ratios, use_ratio = get_mask_ratios(exp_metric)

        violators = 0
        print("test type {0} pert type {1} expl type {2}".format(modality, pert_type, method))
        print("exp_metric {0} delta_type {1}".format(exp_metric, delta_type))

        mask_size = perturbation_arguments.args.mask_size
        b_size = perturbation_arguments.args.b_size


        if use_ratio:
            num_slots = len(mask_ratios)
        else:
            num_slots = (image_boxes_len // mask_size)

        if num_slots>20:
            num_slots = 20

        if exp_metric == 'Violation':
            mask_size = 1
            print('Violation mask size must be 1')

        accs, spccs, weights, nongt_weights, spatials, preds = [torch.tensor((), dtype=torch.float32) for _ in
                                                                       range(6)]
        top_ans_deltas, newpred_accs, all_pred_deltas = [torch.tensor((), dtype=torch.float32) for _ in range(3)]
        step_acc = [0] * num_slots

        if task == "gqa": 
            cls_index = torch.zeros(1).long()
        else:
            cls_index = None
        
        layer_hiddens = {}
        layer_hiddens["num_layers"] = len(self.model.model.bert.encoder.layer)
        for b_idx, batch in enumerate(tqdm.tqdm(loader, disable=disable_tqdm, total=num_samples//b_size)):
            # input_mask = batch['input_mask']
            b_end = (b_idx+1)*b_size
            if b_end > num_samples:
                break

            if not has_cached_exp:
                method_cam = method_expl[method](batch, cls_index=cls_index)
                cached_exp_scores.append(method_cam.detach().clone().cpu().numpy())
                method_cam = method_cam[:, -image_boxes_len:]
            else:
                method_cam = cached_exp_scores[b_idx*b_size :b_end]
                method_cam = method_cam.cuda()

            exp_scores = method_cam

            image_features = batch['image_feature_0'].clone().cuda()
            image_bboxes = torch.from_numpy(np.stack(batch['image_info_0']['bbox'])).clone().cuda()

            # find top step boxess
            if exp_metric == "Sufficiency":  # increasing order
                a1_max, a1_max_ind = exp_scores.neg().topk(k=image_boxes_len, dim=-1)
                a1_max = a1_max.neg()
            elif exp_metric == "RC":
                a1_max, a1_max_ind = exp_scores.abs().topk(k=image_boxes_len, dim=-1)
            elif exp_metric == "Violation":
                _, a1_max_ind = exp_scores.abs().topk(k=image_boxes_len, dim=-1)
                a1_max = exp_scores.gather(dim=1, index=a1_max_ind)
            else:
                a1_max, a1_max_ind = exp_scores.topk(k=image_boxes_len, dim=-1)

            report = self._forward(batch)
            acc = compute_score_with_logits(report["scores"].data, report["targets"])  # b

            # print("original:", acc)
            probs = sigmoid(report["scores"])
            topAns_prob, topAns_pred_ind = probs.topk(k=1, dim=1)

            batch_topAns_deltas, batch_newpred_acc, batch_pred_deltas, batch_weights = \
                [torch.zeros((b_size, num_slots)) for _ in range(4)]
            for step_idx, rank in enumerate(range(num_slots)):
                if use_ratio:
                    ratio = mask_ratios[rank]
                    mask_size = round(ratio * image_boxes_len)  # [b]
                    exp_rank_ind = a1_max_ind[:, :mask_size]  # e.g., top-[5%, 10%, 50%] or last-[95%, 80%]
                    exp_rank_weights = a1_max[:, :mask_size]
                else:
                    exp_rank_ind = a1_max_ind[:, rank * mask_size:mask_size * (rank + 1)]
                    exp_rank_weights = a1_max[:, rank * mask_size:mask_size * (rank + 1)]

                batch_weights[:, rank] = exp_rank_weights.sum(dim=-1).cpu().data

                masks = torch.ones_like(exp_scores)
                masks = masks.scatter(dim=1, index=exp_rank_ind, value=0).long()  # [b, o]
                curr_num_tokens = masks.sum(dim=-1)
                # print(batch['image_info_0']['max_features'], batch['image_info_0']['num_boxes'])

                if delta_type == "zeros_mask":
                    batch['image_feature_0'] = image_features * (masks.unsqueeze(2))
                    batch['image_info_0']['bbox'] = image_bboxes * (masks.unsqueeze(2))
                elif delta_type == "slice_out":
                    # remove the top step boxes from the batch info
                    batch['image_feature_0'] = image_features[masks.bool()].view(-1,
                                                                                 image_boxes_len-mask_size,
                                                                                 2048)
                    batch['image_info_0']['bbox'] = image_bboxes[masks.bool()].view(-1,
                                                                                   image_boxes_len-mask_size,
                                                                                   4)
                    batch['image_info_0']['max_features'] = curr_num_tokens.to(batch['image_feature_0'].device).view(-1)
                    batch['image_info_0']['num_boxes'] = curr_num_tokens.tolist()

                elif delta_type == "att_mask":
                    batch.update({
                        "visual_attention_mask": masks
                    })


                report = self._forward(batch)
                new_probs = sigmoid(report["scores"])
                new_topAns_prob = new_probs.gather(dim=1, index=topAns_pred_ind)  # [b, k]
                batch_topAns_deltas[:, rank] = (topAns_prob - new_topAns_prob).mean(dim=1).detach().cpu()  # b
                batch_pred_deltas[:, rank] = ((probs) - (new_probs)).abs().mean(dim=1).detach().cpu()  # b
                new_acc = compute_score_with_logits(report["scores"].data, report["targets"]) # b
                batch_newpred_acc[:, rank] = new_acc.cpu()
                # step_acc[step_idx] += new_acc.mean().cpu().item()
                # print("step_idx:", new_acc)

            top_ans_deltas = torch.cat([top_ans_deltas, batch_topAns_deltas.detach().cpu()])
            newpred_accs = torch.cat([newpred_accs, batch_newpred_acc.detach().cpu()])
            all_pred_deltas = torch.cat([all_pred_deltas, batch_pred_deltas.detach().cpu()])
            preds = torch.cat([preds, topAns_prob.detach().cpu()])
            weights = torch.cat([weights, batch_weights.detach().cpu()])
            accs = torch.cat([accs, acc.view(-1).detach().cpu()])


        if only_perturbation:
            save_scores_path = os.path.join(exp_path, method)
            create_dir(save_scores_path)
            with open(os.path.join(save_scores_path, exp_metric+delta_type+'.pkl'), 'wb') as f:
                print("save {} perturbation scores to {}".format(exp_metric+delta_type, save_scores_path))
                pickle.dump({
                    "top_ans_deltas": top_ans_deltas.numpy(),
                    "newpred_accs": newpred_accs.numpy(),
                    "all_pred_deltas": all_pred_deltas.numpy(),
                    "preds": preds.numpy(),
                    "weights": weights.numpy(),
                    "accs": accs.numpy(),
                    "mask_ratios": mask_ratios
                }, f)

        if not has_cached_exp:
            with open(cached_exp_path, 'wb') as f:
                # cached_exp_scores = cat(cached_exp_scores, dim=0).numpy()
                print("save {} exp scores to {}".format(method, cached_exp_path))
                pickle.dump(cached_exp_scores, f)

        print(accs.mean())
        print(newpred_accs.mean(dim=0))



