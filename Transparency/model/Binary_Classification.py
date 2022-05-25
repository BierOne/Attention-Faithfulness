import json
import os
import shutil
from copy import deepcopy
from typing import Dict

import pickle
import numpy as np
import torch
import torch.nn as nn
from allennlp.common import Params
from sklearn.utils import shuffle
from tqdm import tqdm
from torch import cat
from os.path import exists


from Transparency.model.modules.Decoder import AttnDecoder
from Transparency.model.modules.Encoder import Encoder

from .modelUtils import BatchHolder, get_sorting_index_with_noise_from_lengths
from .modelUtils import jsd as js_divergence

file_name = os.path.abspath(__file__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AdversaryMulti(nn.Module) :
    def __init__(self, decoder=None, metrics_type='Single_Label') :
        super().__init__()
        self.decoder = decoder
        self.metrics_type = metrics_type
        self.K = 5

    def forward(self, data) :
        data.hidden_volatile = data.hidden.detach()

        new_attn = torch.log(data.generate_uniform_attn()).unsqueeze(1).repeat(1, self.K, 1) #(B, 10, L)
        new_attn = new_attn + torch.randn(new_attn.size()).to(device)*3

        new_attn.requires_grad = True
        
        data.log_attn_volatile = new_attn 
        optim = torch.optim.Adam([data.log_attn_volatile], lr=0.01, amsgrad=True)

        for _ in range(500):
            log_attn = data.log_attn_volatile + 1 - 1
            log_attn.masked_fill_(data.masks.unsqueeze(1).bool(), -float('inf'))
            data.attn_volatile = nn.Softmax(dim=-1)(log_attn) #(B, 10, L)
            self.decoder.get_output(data)
            predict_new = data.predict_volatile #(B, 10, O)

            if self.metrics_type == 'Single_Label':
                y_diff = torch.sigmoid(predict_new) - torch.sigmoid(data.predict.detach()).unsqueeze(1)  # (B, 10, O)
                diff = nn.ReLU()(torch.abs(y_diff).sum(-1, keepdim=True) - 1e-2) #(B, 10, 1)

            else:
                y_diff = self.output_abs_diff(predict_new, data.predict.detach().unsqueeze(1))
                diff = nn.ReLU()(y_diff - 1e-2) #(B, *, 1)

            jsd = js_divergence(data.attn_volatile, data.attn.detach().unsqueeze(1)) #(B, 10, 1)
            cross_jsd = js_divergence(data.attn_volatile.unsqueeze(1), data.attn_volatile.unsqueeze(2))
            
            loss =  -(jsd**1) + 500 * diff
            loss = loss.sum() - cross_jsd.sum(0).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

        log_attn = data.log_attn_volatile + 1 - 1
        log_attn.masked_fill_(data.masks.unsqueeze(1).bool(), -float('inf'))
        data.attn_volatile = nn.Softmax(dim=-1)(log_attn)
        self.decoder.get_output(data)
        data.predict_volatile = torch.sigmoid(data.predict_volatile)

    def output_abs_diff(self, p, q):
        # p : (B, *, O)
        # q : (B, *, O)
        softmax = nn.Softmax(dim=-1)
        y_diff = torch.abs(softmax(p) - softmax(q)).sum(-1).unsqueeze(-1)  # (B, *, 1)

        return y_diff

    def output_diff(self, p, q):
        # p : (B, *, O)
        # q : (B, *, O)
        softmax = nn.Softmax(dim=-1)
        y_diff = (softmax(p) - softmax(q)).sum(-1).unsqueeze(-1)  # (B, *, 1)

        return y_diff


class Model() :
    def __init__(self, configuration, pre_embed=None, metrics_type='Single_Label') :
        configuration = deepcopy(configuration)
        self.configuration = deepcopy(configuration)

        configuration['model']['encoder']['pre_embed'] = pre_embed
        self.encoder = Encoder.from_params(Params(configuration['model']['encoder'])).to(device)

        configuration['model']['decoder']['hidden_size'] = self.encoder.output_size
        self.decoder = AttnDecoder.from_params(Params(configuration['model']['decoder'])).to(device)

        self.encoder_params = list(self.encoder.parameters())
        self.attn_params = list([v for k, v in self.decoder.named_parameters() if 'attention' in k])
        self.decoder_params = list([v for k, v in self.decoder.named_parameters() if 'attention' not in k])

        self.bsize = configuration['training']['bsize']
        
        weight_decay = configuration['training'].get('weight_decay', 1e-5)
        self.encoder_optim = torch.optim.Adam(self.encoder_params, lr=0.001, weight_decay=weight_decay, amsgrad=True)
        self.attn_optim = torch.optim.Adam(self.attn_params, lr=0.001, weight_decay=0, amsgrad=True)
        self.decoder_optim = torch.optim.Adam(self.decoder_params, lr=0.001, weight_decay=weight_decay, amsgrad=True)
        self.adversarymulti = AdversaryMulti(decoder=self.decoder)

        self.metrics_type = metrics_type

        if self.metrics_type == 'Single_Label':
            pos_weight = configuration['training'].get('pos_weight', [1.0] * self.decoder.output_size)
            self.pos_weight = torch.Tensor(pos_weight).to(device)
            self.criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)
        else:
            self.criterion = nn.CrossEntropyLoss()

        import time
        dirname = configuration['training']['exp_dirname']
        basepath = configuration['training'].get('basepath', 'outputs')
        self.time_str = time.ctime().replace(' ', '_')
        self.dirname = os.path.join(basepath, dirname, self.time_str)
        
    # @classmethod
    # def init_from_config(cls, dirname, **kwargs) :
    #     print(dirname)
    #     config = json.load(open(dirname + '/config.json', 'r'))
    #     config.update(kwargs)
    #     obj = cls(config)
    #     obj.load_values(dirname)
    #     return obj
    
    @classmethod
    def init_from_config(cls, dirname, config_update=None, metrics_type='Single_Label') :
        config = json.load(open(dirname + '/config.json', 'r'))
        if config_update is not None:
            config.update(config_update)
        obj = cls(config, metrics_type=metrics_type)
        obj.load_values(dirname)
        return obj

    def train(self, data_in, target_in, train=True) :
        sorting_idx = get_sorting_index_with_noise_from_lengths([len(x) for x in data_in], noise_frac=0.1)
        data = [data_in[i] for i in sorting_idx]
        target = [target_in[i] for i in sorting_idx]
        
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)
        loss_total = 0

        batches = list(range(0, N, bsize))
        batches = shuffle(batches)

        for n in tqdm(batches) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_target = target[n:n+bsize]

            if self.metrics_type == 'Single_Label':
                batch_target = torch.Tensor(batch_target).to(device)
                if len(batch_target.shape) == 1:  # (B, )
                    batch_target = batch_target.unsqueeze(-1)  # (B, 1)
                bce_loss = self.criterion(batch_data.predict, batch_target)
                weight = batch_target * self.pos_weight + (1 - batch_target)
                bce_loss = (bce_loss * weight).mean(1).sum()
                loss = bce_loss
            else:
                batch_target = torch.LongTensor(batch_target).to(device)
                ce_loss = self.criterion(batch_data.predict, batch_target)
                loss = ce_loss


            if hasattr(batch_data, 'reg_loss') :
                loss += batch_data.reg_loss

            if train :
                self.encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()
                self.attn_optim.zero_grad()
                loss.backward()
                self.encoder_optim.step()
                self.decoder_optim.step()
                self.attn_optim.step()

            loss_total += float(loss.data.cpu().item())
        return loss_total*bsize/N

    def evaluate(self, data) :
        self.encoder.eval()
        self.decoder.eval()
        bsize = self.bsize
        N = len(data)

        outputs = []
        attns = []

        for n in tqdm(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            if self.metrics_type == 'Single_Label':
                batch_data.predict = torch.sigmoid(batch_data.predict)
            else:
                batch_data.predict = torch.argmax(batch_data.predict, dim=-1)

            if self.decoder.use_attention :
                attn = batch_data.attn.cpu().data.numpy()
                attns.append(attn)

            predict = batch_data.predict.cpu().data.numpy()
            outputs.append(predict)

        outputs = [x for y in outputs for x in y]
        if self.decoder.use_attention :
            attns = [x for y in attns for x in y]
        
        return outputs, attns 

    def replace_and_run(self, data, exp="att", delta_type="att_mask", exp_metric="AUCTP", folds=10, mask_size=1,
                        logits_force=False, load_exp_scores=False):
        def _get_prob(logits, metric_type, retain_grad=False):
            if not retain_grad:
                logits = logits.detach().data
            if metric_type == 'Single_Label':
                prob = torch.sigmoid(logits)
                if prob.shape[-1] == 1:
                    prob = torch.cat([1 - prob, prob], 1)
                pred_label = prob.detach().data
            else:
                prob = softmax(logits)
                pred_label = torch.argmax(prob.detach().data, dim=-1)
            return prob, pred_label.cpu()


        def _get_mask_ratios(exp_metric):
            use_ratio = True
            if exp_metric == "AUCTP": # include completeness
                mask_ratios = np.linspace(0, 90, num=10)
                mask_ratios[0] = 5 # array([0.05, 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 ])
                mask_ratios = mask_ratios / 100
            elif exp_metric == "Sufficiency": # include completeness
                mask_ratios = np.array([0.5, 0.8, 0.9, 0.95]) # increasing order
            elif exp_metric == "Comprehensiveness": # include completeness
                mask_ratios = np.array([0.05, 0.1, 0.2, 0.5]) # decending order
            else:
                mask_ratios = None
                use_ratio = False

            return mask_ratios, use_ratio

        mask_ratios, use_ratio = _get_mask_ratios(exp_metric)
        self.encoder.eval()
        self.decoder.eval()
        bsize = self.bsize
        softmax = nn.Softmax(dim=-1)

        has_cached_exp, cached_exp_scores = False, []
        cached_exp_path = os.path.join(self.dirname, exp+"_bz{}.pkl".format(bsize))

        if exists(cached_exp_path) and load_exp_scores:
            print("load exp scores from {}".format(cached_exp_path))
            cached_exp_scores = pickle.load(open(cached_exp_path, 'rb'))
            has_cached_exp = True

        N = len(data)
        preds, new_preds = [torch.tensor((), dtype=torch.float32) for _ in range(2)]
        top_ans_deltas, weights, all_pred_deltas = [], [], []
        min_len = 32

        if exp_metric == 'Violation':
            mask_size = 1
            print('Violation mask size must be 1')

        for b_idx, n in enumerate(tqdm(range(0, N, bsize))):
            torch.cuda.empty_cache()
            batch_doc = data[n:n + bsize]
            batch_data = BatchHolder(batch_doc)
            batch_data.keep_grads = True
            if batch_data.lengths.min() < min_len:
                min_len = batch_data.lengths.min() # generally, minlength is 32 in the dataset

            if use_ratio:
                num_slots = len(mask_ratios)
            else:
                num_slots = (min_len/ mask_size).long()

            if num_slots > 20:
                num_slots = 20

            self.encoder(batch_data)
            self.decoder(batch_data)

            attn = batch_data.attn
            pred = batch_data.predict
            prob, pred_label = _get_prob(logits=pred, metric_type=self.metrics_type, retain_grad=True)
            preds = cat([preds, pred_label.detach().cpu()])

            topAns_prob, topAns_pred_ind = prob.detach().topk(k=1, dim=1)

            if not has_cached_exp:
                ################################ Calculate Explanation Scores #####################################
                if exp in ['grad_cam'] or logits_force:
                    backward_output = pred # logits
                else:
                    backward_output = prob

                if self.metrics_type == 'Single_Label':
                    # here we use the sign since model only output one score which, however,
                    # represents two labels (False: 1-p, True: p). As such, two scores should 
                    # have different signs (-p and p), denoting contrary model tendency.
                    logits_weight = (topAns_pred_ind - 0.5).sign().data # 1 for True, -1 for False
                    backward_output = backward_output * (logits_weight)

                backward_output = backward_output.gather(dim=1, index=topAns_pred_ind)  # [b, k]

                em = batch_data.embedding
                if exp == "grad_cam": # backpropagate with logits
                    grad_cam = torch.autograd.grad(backward_output.sum(), em, create_graph=False, retain_graph=False)[0]
                    exp_scores = grad_cam.sum(-1).data  # B x num_tokens

                elif exp == "input*grad":
                    grad = torch.autograd.grad(backward_output.sum(), em, create_graph=False, retain_graph=False)[0]
                    exp_scores = (grad * em).sum(-1).data # B x num_tokens

                elif "att*grad" in exp:
                    att_grad = torch.autograd.grad(backward_output.sum(), attn, create_graph=False, retain_graph=False)[0]
                    if "sign" in exp:
                        fused_att = torch.sign(att_grad) * attn  # [b, num_heads, q_len, num_obj]
                    elif "only_grad" in exp:
                        fused_att = att_grad  # [b, o, g, 1] or [b, num_heads, q_len, num_obj]
                    elif "grad_abs" in exp:
                        fused_att = att_grad.abs() * attn # [b, o, g, 1] or [b, num_heads, q_len, num_obj]
                    else:
                        fused_att = att_grad * attn  # [b, o, g, 1] or [b, num_heads, q_len, num_obj]
                    exp_scores = fused_att.data

                elif exp == "ig":
                    grad = torch.autograd.grad(backward_output.sum(), em, create_graph=False, retain_graph=True)[0]
                    for i in range(1, folds):
                        ig_ratio = i/folds
                        batch_data = BatchHolder(batch_doc)
                        batch_data.keep_grads = True
                        self.encoder(batch_data, ig_ratio)
                        self.decoder(batch_data)
                        new_prob, _ = _get_prob(logits=batch_data.predict, metric_type=self.metrics_type, retain_grad=True)
                        if self.metrics_type == 'Single_Label':
                            # here we select the same label, and accumulate the corresponding gradients
                            backward_output = new_prob * logits_weight
                            backward_output = backward_output.gather(dim=1, index=topAns_pred_ind)  # [b, k]
                        else:
                            backward_output = new_prob.gather(dim=1, index=topAns_pred_ind)  # [b, k]
                        grad += torch.autograd.grad(backward_output.sum(), batch_data.embedding, create_graph=False, retain_graph=True)[0]
                    exp_scores = (grad * em).sum(-1).data  # B x num_tokens

                elif exp == "rand":
                    exp_scores = torch.rand_like(attn.data)

                elif exp == "att_norm":
                    exp_scores = self.decoder.get_att_norm(batch_data)
                else:
                    exp_scores = attn.data
                cached_exp_scores.append(exp_scores.cpu().numpy()) # B x num_tokens


            else:
                # if cached, we directly load exp scores
                exp_scores = torch.tensor(cached_exp_scores[b_idx]).cuda()

            if exp_metric == "Sufficiency": # increasing order
                a1_max, a1_max_ind = exp_scores.neg().topk(k=batch_data.maxlen, dim=1)
                a1_max = a1_max.neg()
            elif exp_metric == "RC":
                a1_max, a1_max_ind = exp_scores.abs().topk(k=batch_data.maxlen, dim=-1)
            elif exp_metric == "Violation":
                _, a1_max_ind = exp_scores.abs().topk(k=batch_data.maxlen, dim=-1)
                a1_max = exp_scores.gather(dim=1, index=a1_max_ind)
            else:
                a1_max, a1_max_ind = exp_scores.topk(k=batch_data.maxlen, dim=1)

            prob = prob.detach().data
            batch_pred_deltas = torch.zeros((batch_data.B, num_slots)).cuda()
            batch_topAns_deltas = torch.zeros((batch_data.B, num_slots)).cuda()
            batch_weights = torch.zeros((batch_data.B, num_slots))
            if self.metrics_type == 'Single_Label':
                batch_pred_label = torch.zeros((batch_data.B, num_slots, 2)) # soft probability
            else:
                batch_pred_label = torch.zeros((batch_data.B, num_slots))
            ################################ Calculate Replacement Scores #####################################
            zeros_mask = None
            for rank in range(num_slots):
                batch_data = BatchHolder(batch_doc)
                if use_ratio:
                    # batch_data = BatchHolder(batch_doc, mask_ratio=mask_ratios[rank], exp_socre=a1_max.cpu(),
                    #                          exp_socre_ind=a1_max_ind.cpu(), delta_type=delta_type)
                    # batch_weights[:, rank] = batch_data.mask_weights
                    ratio = mask_ratios[rank]
                    mask_size = (ratio * batch_data.lengths) # [b]
                    mask_size = mask_size.min().long()

                    mask_size = mask_size+1 if mask_size == 0 else mask_size
                    # unique_sizes, counts = mask_size.unique(sorted=True, return_counts=True)
                    # print(unique_sizes, counts)
                    exp_rank_ind = a1_max_ind[:, :mask_size] # e.g., top-[5%, 10%, 50%] or last-[95%, 80%]
                    exp_rank_weights = a1_max[:, :mask_size]

                else:
                    exp_rank_ind = a1_max_ind[:, rank * mask_size:mask_size * (rank + 1)]
                    exp_rank_weights = a1_max[:, rank * mask_size:mask_size * (rank + 1)]

                batch_weights[:, rank] = exp_rank_weights.sum(dim=-1).cpu().data
                masks = torch.ones_like(attn)
                masks = masks.scatter(dim=1, index=exp_rank_ind, value=0)  # [b, o], 1->no_mask, 0->mask
                if delta_type == "slice_out":
                    # print(ratio, batch_data.seq.shape, mask_size, batch_data.maxlen-mask_size)
                    batch_data.seq = batch_data.seq[masks.bool()].view(-1, batch_data.maxlen-mask_size)
                    batch_data.lengths = batch_data.lengths - mask_size
                    batch_data.masks = batch_data.masks[masks.bool()].view(-1, (batch_data.maxlen)-mask_size)
                elif delta_type == "att_mask":
                    # mask attention + re-normalize
                    batch_data.masks = (batch_data.masks.bool() | (~masks.bool())).float()
                elif delta_type == "zeros_mask": # MASK TOKEN ([MASK] is the padding)
                    zeros_mask = masks

                self.encoder(batch_data, zeros_mask=zeros_mask)
                self.decoder(batch_data)
                new_prob, new_pred_label = _get_prob(logits=batch_data.predict, metric_type=self.metrics_type)
                new_topAns_prob = new_prob.gather(dim=1, index=topAns_pred_ind)  # [b, 1]

                # print(ratio, (~batch_data.masks).sum(dim=-1)[0], batch_data.predict[0], new_pred_label[0])
                batch_topAns_deltas[:, rank] = (topAns_prob - new_topAns_prob).sum(dim=1) # [b]
                batch_pred_deltas[:, rank] = (prob - new_prob).abs().mean(dim=-1)
                batch_pred_label[:, rank] = new_pred_label

            top_ans_deltas.append(batch_topAns_deltas.detach().cpu())
            all_pred_deltas.append(batch_pred_deltas.detach().cpu())
            weights.append(batch_weights.detach().cpu())
            new_preds = cat([new_preds, batch_pred_label.detach().cpu()])

        if not has_cached_exp:
            with open(cached_exp_path, 'wb') as f:
                pickle.dump(cached_exp_scores, f)
        preds = preds.cpu().numpy()
        new_preds = new_preds.cpu().numpy()
        weights = cat(weights, dim=0)
        top_ans_deltas = cat(top_ans_deltas, dim=0)
        all_pred_deltas = cat(all_pred_deltas, dim=0)
        # print(new_preds.shape, new_preds)

        return preds, new_preds, weights, top_ans_deltas, all_pred_deltas, mask_ratios


    def save_values(self, use_dirname=None, save_model=True) :

        print ('saved config ',self.configuration)

        if use_dirname is not None :
            dirname = use_dirname
        else:
            dirname = self.dirname
        os.makedirs(dirname, exist_ok=True)
        shutil.copy2(file_name, dirname + '/')
        json.dump(self.configuration, open(dirname + '/config.json', 'w'))

        if save_model :
            torch.save(self.encoder.state_dict(), dirname + '/enc.th')
            torch.save(self.decoder.state_dict(), dirname + '/dec.th')

        return dirname

    def load_values(self, dirname) :
        self.encoder.load_state_dict(torch.load(dirname + '/enc.th', lambda storage, loc: storage.cuda()))
        self.decoder.load_state_dict(torch.load(dirname + '/dec.th', lambda storage, loc: storage.cuda()))





    # def gradient_mem(self, data) :
    #     self.encoder.train()
    #     self.decoder.train()
    #     bsize = self.bsize
    #     N = len(data)

    #     grads = {'XxE' : [], 'XxE[X]' : [], 'H' : []}

    #     for n in tqdm(range(0, N, bsize)) :
    #         torch.cuda.empty_cache()
    #         batch_doc = data[n:n+bsize]

    #         grads_xxe = []
    #         grads_xxex = []
    #         grads_H = []
            
    #         for i in range(self.decoder.output_size) :
    #             batch_data = BatchHolder(batch_doc)
    #             batch_data.keep_grads = True
    #             batch_data.detach = True

    #             self.encoder(batch_data) 
    #             self.decoder(batch_data)

    #             if self.metrics_type == 'Single_Label':
    #                 torch.sigmoid(batch_data.predict[:, i]).sum().backward()
    #             else:
    #                 max_predict = torch.argmax(batch_data.predict, dim=-1)
    #                 prob_predict = nn.Softmax(dim=-1)(batch_data.predict)

    #                 max_class_prob = torch.gather(prob_predict, -1, max_predict.unsqueeze(-1))
    #                 max_class_prob.sum().backward()

    #             g = batch_data.embedding.grad
    #             em = batch_data.embedding
    #             g1 = (g * em).sum(-1)
                
    #             grads_xxex.append(g1.cpu().data.numpy())
                
    #             g1 = (g * self.encoder.embedding.weight.sum(0)).sum(-1)
    #             grads_xxe.append(g1.cpu().data.numpy())
                
    #             g1 = batch_data.hidden.grad.sum(-1)
    #             grads_H.append(g1.cpu().data.numpy())

    #         grads_xxe = np.array(grads_xxe).swapaxes(0, 1)
    #         grads_xxex = np.array(grads_xxex).swapaxes(0, 1)
    #         grads_H = np.array(grads_H).swapaxes(0, 1)

    #         grads['XxE'].append(grads_xxe)
    #         grads['XxE[X]'].append(grads_xxex)
    #         grads['H'].append(grads_H)

    #     for k in grads :
    #         grads[k] = [x for y in grads[k] for x in y]
                    
    #     return grads       
    
    # def remove_and_run(self, data) :
    #     self.encoder.train()
    #     self.decoder.train()
    #     bsize = self.bsize
    #     N = len(data)

    #     outputs = []

    #     for n in tqdm(range(0, N, bsize)) :
    #         batch_doc = data[n:n+bsize]
    #         batch_data = BatchHolder(batch_doc)
    #         po = np.zeros((batch_data.B, batch_data.maxlen, self.decoder.output_size))

    #         for i in range(1, batch_data.maxlen - 1) :
    #             batch_data = BatchHolder(batch_doc)

    #             batch_data.seq = torch.cat([batch_data.seq[:, :i], batch_data.seq[:, i+1:]], dim=-1)
    #             batch_data.lengths = batch_data.lengths - 1
    #             batch_data.masks = torch.cat([batch_data.masks[:, :i], batch_data.masks[:, i+1:]], dim=-1)

    #             self.encoder(batch_data)
    #             self.decoder(batch_data)

    #             if self.metrics_type == 'Single_Label':
    #                 po[:, i] = torch.sigmoid(batch_data.predict).cpu().data.numpy()
    #             else:
    #                 predict_difference = self.adversary_multi.output_abs_diff(batch_data_loop.predict, batch_data.predict)
    #                 po[:, i] = predict_difference.squeeze(-1).cpu().data.numpy()

    #         outputs.append(po)

    #     outputs = [x for y in outputs for x in y]
                    
    #     return outputs
    
    # def permute_attn(self, data, num_perm=100) :
    #     self.encoder.train()
    #     self.decoder.train()
    #     bsize = self.bsize
    #     N = len(data)

    #     permutations = []

    #     for n in tqdm(range(0, N, bsize)) :
    #         torch.cuda.empty_cache()
    #         batch_doc = data[n:n+bsize]
    #         batch_data = BatchHolder(batch_doc)

    #         batch_perms = np.zeros((batch_data.B, num_perm, self.decoder.output_size))

    #         self.encoder(batch_data)
    #         self.decoder(batch_data)
            
    #         for i in range(num_perm) :
    #             batch_data.permute = True
    #             self.decoder(batch_data)
    #             output = torch.sigmoid(batch_data.predict)
    #             batch_perms[:, i] = output.cpu().data.numpy()

    #         permutations.append(batch_perms)

    #     permutations = [x for y in permutations for x in y]
                    
    #     return permutations


    # def adversarial_multi(self, data) :
    #     self.encoder.eval()
    #     self.decoder.eval()

    #     for p in self.encoder.parameters() :
    #         p.requires_grad = False

    #     for p in self.decoder.parameters() :
    #         p.requires_grad = False

    #     bsize = self.bsize
    #     N = len(data)

    #     adverse_attn = []
    #     adverse_output = []

    #     for n in tqdm(range(0, N, bsize)) :
    #         torch.cuda.empty_cache()
    #         batch_doc = data[n:n+bsize]
    #         batch_data = BatchHolder(batch_doc)

    #         self.encoder(batch_data)
    #         self.decoder(batch_data)

    #         self.adversarymulti(batch_data)

    #         attn_volatile = batch_data.attn_volatile.cpu().data.numpy() #(B, 10, L)
    #         predict_volatile = batch_data.predict_volatile.cpu().data.numpy() #(B, 10, O)

    #         adverse_attn.append(attn_volatile)
    #         adverse_output.append(predict_volatile)

    #     adverse_output = [x for y in adverse_output for x in y]
    #     adverse_attn = [x for y in adverse_attn for x in y]
        
    #     return adverse_output, adverse_attn

    # def logodds_attention(self, data, logodds_map:Dict) :
    #     self.encoder.eval()
    #     self.decoder.eval()

    #     bsize = self.bsize
    #     N = len(data)

    #     adverse_attn = []
    #     adverse_output = []

    #     logodds = np.zeros((self.encoder.vocab_size, ))
    #     for k, v in logodds_map.items() :
    #         if v is not None :
    #             logodds[k] = abs(v)
    #         else :
    #             logodds[k] = float('-inf')
    #     logodds = torch.Tensor(logodds).to(device)

    #     for n in tqdm(range(0, N, bsize)) :
    #         torch.cuda.empty_cache()
    #         batch_doc = data[n:n+bsize]
    #         batch_data = BatchHolder(batch_doc)

    #         self.encoder(batch_data)
    #         self.decoder(batch_data)

    #         attn = batch_data.attn #(B, L)
    #         batch_data.attn_logodds = logodds[batch_data.seq]
    #         self.decoder.get_output_from_logodds(batch_data)

    #         attn_volatile = batch_data.attn_volatile.cpu().data.numpy() #(B, L)
    #         predict_volatile = torch.sigmoid(batch_data.predict_volatile).cpu().data.numpy() #(B, O)

    #         adverse_attn.append(attn_volatile)
    #         adverse_output.append(predict_volatile)

    #     adverse_output = [x for y in adverse_output for x in y]
    #     adverse_attn = [x for y in adverse_attn for x in y]
        
    #     return adverse_output, adverse_attn

    # def logodds_substitution(self, data, top_logodds_words:Dict) :
    #     self.encoder.eval()
    #     self.decoder.eval()

    #     bsize = self.bsize
    #     N = len(data)

    #     adverse_X = []
    #     adverse_attn = []
    #     adverse_output = []

    #     words_neg = torch.Tensor(top_logodds_words[0][0]).long().cuda().unsqueeze(0)
    #     words_pos = torch.Tensor(top_logodds_words[0][1]).long().cuda().unsqueeze(0)

    #     words_to_select = torch.cat([words_neg, words_pos], dim=0) #(2, 5)

    #     for n in tqdm(range(0, N, bsize)) :
    #         torch.cuda.empty_cache()
    #         batch_doc = data[n:n+bsize]
    #         batch_data = BatchHolder(batch_doc)

    #         self.encoder(batch_data)
    #         self.decoder(batch_data)
    #         predict_class = (torch.sigmoid(batch_data.predict).squeeze(-1) > 0.5)*1 #(B,)

    #         attn = batch_data.attn #(B, L)
    #         top_val, top_idx = torch.topk(attn, 5, dim=-1)
    #         subs_words = words_to_select[1 - predict_class.long()] #(B, 5)

    #         batch_data.seq.scatter_(1, top_idx, subs_words)

    #         self.encoder(batch_data)
    #         self.decoder(batch_data)

    #         attn_volatile = batch_data.attn.cpu().data.numpy() #(B, L)
    #         predict_volatile = torch.sigmoid(batch_data.predict).cpu().data.numpy() #(B, O)
    #         X_volatile = batch_data.seq.cpu().data.numpy()

    #         adverse_X.append(X_volatile)
    #         adverse_attn.append(attn_volatile)
    #         adverse_output.append(predict_volatile)

    #     adverse_X = [x for y in adverse_X for x in y]
    #     adverse_output = [x for y in adverse_output for x in y]
    #     adverse_attn = [x for y in adverse_attn for x in y]
        
    #     return adverse_output, adverse_attn, adverse_X
