import json
import os
import shutil
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import shuffle
from tqdm import tqdm
from allennlp.common import Params

import pickle
from torch import cat
from os.path import exists

from .modelUtils import isTrue, get_sorting_index_with_noise_from_lengths
from .modelUtils import BatchHolder, BatchMultiHolder

from Transparency.model.modules.Decoder import AttnDecoderQA
from Transparency.model.modules.Encoder import Encoder

from .modelUtils import jsd as js_divergence

file_name = os.path.abspath(__file__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AdversaryMulti(nn.Module) :
    def __init__(self, decoder=None) :
        super().__init__()
        self.decoder = decoder
        self.K = 5

    def forward(self, data) :
        data.P.hidden_volatile = data.P.hidden.detach()
        data.Q.last_hidden_volatile = data.Q.last_hidden.detach()

        new_attn = torch.log(data.P.generate_uniform_attn()).unsqueeze(1).repeat(1, self.K, 1) #(B, 10, L)
        new_attn = new_attn + torch.randn(new_attn.size()).to(device)*3

        new_attn.requires_grad = True
        
        data.log_attn_volatile = new_attn 
        optim = torch.optim.Adam([data.log_attn_volatile], lr=0.01, amsgrad=True)
        data.multiattention = True

        for _ in range(500) :
            log_attn = data.log_attn_volatile + 1 - 1
            log_attn.masked_fill_(data.P.masks.unsqueeze(1).bool(), -float('inf'))
            data.attn_volatile = nn.Softmax(dim=-1)(log_attn) #(B, 10, L)
            self.decoder.get_output(data)
            
            predict_new = data.predict_volatile #(B, *, O)
            y_diff = self.output_abs_diff(predict_new, data.predict.detach().unsqueeze(1))
            diff = nn.ReLU()(y_diff - 1e-2) #(B, *, 1)

            jsd = js_divergence(data.attn_volatile, data.attn.detach().unsqueeze(1)) #(B, *, 1)

            cross_jsd = js_divergence(data.attn_volatile.unsqueeze(1), data.attn_volatile.unsqueeze(2))

            loss =  -(jsd**1) + 500 * diff #(B, *, 1)
            loss = loss.sum() - cross_jsd.sum(0).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()


        log_attn = data.log_attn_volatile + 1 - 1
        log_attn.masked_fill_(data.P.masks.unsqueeze(1).bool(), -float('inf'))
        data.attn_volatile = nn.Softmax(dim=-1)(log_attn)
        self.decoder.get_output(data)

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
    def __init__(self, configuration, pre_embed=None, ) :
        configuration = deepcopy(configuration)
        self.configuration = deepcopy(configuration)

        configuration['model']['encoder']['pre_embed'] = pre_embed

        encoder_copy = deepcopy(configuration['model']['encoder'])
        self.Pencoder = Encoder.from_params(Params(configuration['model']['encoder'])).to(device)
        self.Qencoder = Encoder.from_params(Params(encoder_copy)).to(device)

        configuration['model']['decoder']['hidden_size'] = self.Pencoder.output_size
        self.decoder = AttnDecoderQA.from_params(Params(configuration['model']['decoder'])).to(device)

        self.bsize = configuration['training']['bsize']

        self.adversary_multi = AdversaryMulti(self.decoder)

        weight_decay = configuration['training'].get('weight_decay', 1e-5)
        self.params = list(self.Pencoder.parameters()) + list(self.Qencoder.parameters()) + list(self.decoder.parameters())
        self.optim = torch.optim.Adam(self.params, weight_decay=weight_decay, amsgrad=True)
        # self.optim = torch.optim.Adagrad(self.params, lr=0.05, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        import time
        dirname = configuration['training']['exp_dirname']
        basepath = configuration['training'].get('basepath', 'outputs')
        self.time_str = time.ctime().replace(' ', '_')
        self.dirname = os.path.join(basepath, dirname, self.time_str)

    @classmethod
    def init_from_config(cls, dirname, **kwargs) :
        config = json.load(open(dirname + '/config.json', 'r'))
        config.update(kwargs)
        obj = cls(config)
        obj.load_values(dirname)
        return obj

    def train(self, train_data, train=True) :
        docs_in = train_data.P
        question_in = train_data.Q
        entity_masks_in = train_data.E
        target_in = train_data.A

        sorting_idx = get_sorting_index_with_noise_from_lengths([len(x) for x in docs_in], noise_frac=0.1)
        docs = [docs_in[i] for i in sorting_idx]
        questions = [question_in[i] for i in sorting_idx]
        entity_masks = [entity_masks_in[i] for i in sorting_idx]
        target = [target_in[i] for i in sorting_idx]
        
        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()

        bsize = self.bsize
        N = len(questions)
        loss_total = 0

        batches = list(range(0, N, bsize))
        batches = shuffle(batches)

        for n in tqdm(batches) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = BatchHolder(batch_doc)
            batch_ques = BatchHolder(batch_ques)

            batch_data = BatchMultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

            self.Pencoder(batch_data.P)
            self.Qencoder(batch_data.Q)
            self.decoder(batch_data)

            batch_target = target[n:n+bsize]
            batch_target = torch.LongTensor(batch_target).to(device)

            ce_loss = self.criterion(batch_data.predict, batch_target)

            loss = ce_loss

            if hasattr(batch_data, 'reg_loss') :
                loss += batch_data.reg_loss

            if train :
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            loss_total += float(loss.data.cpu().item())
        return loss_total*bsize/N

    def evaluate(self, data) :
        docs = data.P
        questions = data.Q
        entity_masks = data.E
        
        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()
        
        bsize = self.bsize
        N = len(questions)

        outputs = []
        attns = []
        for n in tqdm(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = BatchHolder(batch_doc)
            batch_ques = BatchHolder(batch_ques)

            batch_data = BatchMultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

            self.Pencoder(batch_data.P)
            self.Qencoder(batch_data.Q)
            self.decoder(batch_data)

            batch_data.predict = torch.argmax(batch_data.predict, dim=-1)
            if self.decoder.use_attention :
                attn = batch_data.attn
                attns.append(attn.cpu().data.numpy())

            predict = batch_data.predict.cpu().data.numpy()
            outputs.append(predict)
            
            

        outputs = [x for y in outputs for x in y]
        attns = [x for y in attns for x in y]
        
        return outputs, attns

    def save_values(self, use_dirname=None, save_model=True) :
        print ('saved config ',self.configuration)

        if use_dirname is not None :
            dirname = use_dirname
        else :
            dirname = self.dirname
        os.makedirs(dirname, exist_ok=True)
        shutil.copy2(file_name, dirname + '/')
        json.dump(self.configuration, open(dirname + '/config.json', 'w'))

        if save_model :
            torch.save(self.Pencoder.state_dict(), dirname + '/encP.th')
            torch.save(self.Qencoder.state_dict(), dirname + '/encQ.th')
            torch.save(self.decoder.state_dict(), dirname + '/dec.th')

        return dirname

    def load_values(self, dirname) :
        self.Pencoder.load_state_dict(torch.load(dirname + '/encP.th'))
        self.Qencoder.load_state_dict(torch.load(dirname + '/encQ.th'))
        self.decoder.load_state_dict(torch.load(dirname + '/dec.th'))


    def replace_and_run(self, data, exp="att", delta_type="att_mask", exp_metric="AUCTP", folds=10, mask_size=1,
                        logits_force=False, load_exp_scores=False):
        def _get_prob(logits, retain_grad=False):
            if not retain_grad:
                logits = logits.detach().data
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
        docs = data.P
        questions = data.Q
        entity_masks = data.E

        self.Pencoder.eval()
        self.Qencoder.eval()
        self.decoder.eval()

        bsize = self.bsize
        softmax = nn.Softmax(dim=-1)

        has_cached_exp, cached_exp_scores = False, []
        cached_exp_path = os.path.join(self.dirname, exp+"_bz{}.pkl".format(bsize))

        if exists(cached_exp_path) and load_exp_scores:
            print("load exp scores from {}".format(cached_exp_path))
            cached_exp_scores = pickle.load(open(cached_exp_path, 'rb'))
            has_cached_exp = True

        N = len(questions)
        preds, new_preds = [torch.tensor((), dtype=torch.float32) for _ in range(2)]
        top_ans_deltas, weights, all_pred_deltas = [], [], []
        min_len = 32


        if exp_metric == 'Violation':
            mask_size = 1
            print('Violation mask size must be 1')

        for b_idx, n in enumerate(tqdm(range(0, N, bsize))):
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_data = BatchMultiHolder(P=BatchHolder(batch_doc), Q=BatchHolder(batch_ques))
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

            batch_data.P.keep_grads = True
            if batch_data.P.lengths.min() < min_len:
                min_len = batch_data.P.lengths.min() # generally, minlength is 32 in the dataset

            if use_ratio:
                num_slots = len(mask_ratios)
            else:
                num_slots = (min_len/ mask_size).long()

            self.Pencoder(batch_data.P)
            self.Qencoder(batch_data.Q)
            self.decoder(batch_data)

            attn = batch_data.attn
            pred = batch_data.predict
            prob, pred_label = _get_prob(logits=pred, retain_grad=True)
            preds = cat([preds, pred_label.detach().cpu()])

            topAns_prob, topAns_pred_ind = prob.detach().topk(k=1, dim=1)

            if not has_cached_exp:
                ################################ Calculate Explanation Scores #####################################
                if exp in ['grad_cam'] or logits_force:
                    backward_output = pred # logits
                else:
                    backward_output = prob
                backward_output = backward_output.gather(dim=1, index=topAns_pred_ind)  # [b, k]

                em = batch_data.P.embedding
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
                    else:
                        fused_att = att_grad * attn  # [b, o, g, 1] or [b, num_heads, q_len, num_obj]
                    exp_scores = fused_att.data

                elif exp == "ig":
                    grad = torch.autograd.grad(backward_output.sum(), em, create_graph=False, retain_graph=True)[0]
                    for i in range(1, folds):
                        ig_ratio = i/folds
                        batch_data = BatchMultiHolder(P=BatchHolder(batch_doc), Q=BatchHolder(batch_ques))
                        batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)
                        batch_data.P.keep_grads = True
                        self.Pencoder(batch_data.P, ig_ratio)
                        self.Qencoder(batch_data.Q)
                        self.decoder(batch_data)
                        new_prob, _ = _get_prob(logits=batch_data.predict, retain_grad=True)
                        backward_output = new_prob.gather(dim=1, index=topAns_pred_ind)  # [b, k]
                        grad += torch.autograd.grad(backward_output.sum(), batch_data.P.embedding, create_graph=False, retain_graph=True)[0]
                    exp_scores = (grad * em).sum(-1).data  # B x num_tokens

                elif exp == "rand":
                    exp_scores = torch.rand_like(attn.data)

                elif exp == "att_norm":
                    exp_scores = self.decoder.get_att_norm(batch_data)
                else:
                    exp_scores = attn.data
                cached_exp_scores.append(exp_scores.cpu().numpy()) # B x num_tokens

            else:
                exp_scores = torch.tensor(cached_exp_scores[b_idx]).cuda()

            if exp_metric == "Sufficiency": # increasing order
                a1_max, a1_max_ind = exp_scores.neg().topk(k=batch_data.P.maxlen, dim=1)
                a1_max = a1_max.neg()
            elif exp_metric == "RC":
                a1_max, a1_max_ind = exp_scores.abs().topk(k=batch_data.P.maxlen, dim=-1)
            elif exp_metric == "Violation":
                _, a1_max_ind = exp_scores.abs().topk(k=batch_data.P.maxlen, dim=-1)
                a1_max = exp_scores.gather(dim=1, index=a1_max_ind)
            else:
                a1_max, a1_max_ind = exp_scores.topk(k=batch_data.P.maxlen, dim=1)

            prob = prob.detach().data
            batch_pred_deltas = torch.zeros((batch_data.P.B, num_slots)).cuda()
            batch_topAns_deltas = torch.zeros((batch_data.P.B, num_slots)).cuda()
            batch_weights = torch.zeros((batch_data.P.B, num_slots))
            batch_pred_label = torch.zeros((batch_data.P.B, num_slots))
            ################################ Calculate Replacement Scores #####################################
            zeros_mask = None
            for rank in range(num_slots):
                batch_data = BatchMultiHolder(P=BatchHolder(batch_doc), Q=BatchHolder(batch_ques))

                batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)
                if use_ratio:
                    # batch_data = BatchHolder(batch_doc, mask_ratio=mask_ratios[rank], exp_socre=a1_max.cpu(),
                    #                          exp_socre_ind=a1_max_ind.cpu(), delta_type=delta_type)
                    # batch_weights[:, rank] = batch_data.mask_weights
                    ratio = mask_ratios[rank]
                    mask_size = (ratio * batch_data.P.lengths) # [b]
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
                    batch_data.P.seq = batch_data.P.seq[masks.bool()].view(-1, batch_data.P.maxlen-mask_size)
                    batch_data.P.lengths = batch_data.P.lengths - mask_size
                    batch_data.P.masks = batch_data.P.masks[masks.bool()].view(-1, (batch_data.P.maxlen)-mask_size)
                elif delta_type == "att_mask":
                    # mask attention + re-normalize
                    batch_data.P.masks = (batch_data.P.masks.bool() | (~masks.bool())).float()
                elif delta_type == "zeros_mask": # MASK TOKEN
                    zeros_mask = masks

                self.Pencoder(batch_data.P, zeros_mask=zeros_mask)
                self.Qencoder(batch_data.Q)
                self.decoder(batch_data)
                new_prob, new_pred_label = _get_prob(logits=batch_data.predict)
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



    # def permute_attn(self, data, num_perm=100) :
    #     docs = data.P
    #     questions = data.Q
    #     entity_masks = data.E

    #     self.Pencoder.train()
    #     self.Qencoder.train()
    #     self.decoder.train()

    #     bsize = self.bsize
    #     N = len(questions)

    #     permutations_predict = []
    #     permutations_diff = []

    #     for n in tqdm(range(0, N, bsize)) :
    #         torch.cuda.empty_cache()
    #         batch_doc = docs[n:n+bsize]
    #         batch_ques = questions[n:n+bsize]
    #         batch_entity_masks = entity_masks[n:n+bsize]

    #         batch_doc = BatchHolder(batch_doc)
    #         batch_ques = BatchHolder(batch_ques)

    #         batch_data = BatchMultiHolder(P=batch_doc, Q=batch_ques)
    #         batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

    #         self.Pencoder(batch_data.P)
    #         self.Qencoder(batch_data.Q)
    #         self.decoder(batch_data)

    #         predict_true = batch_data.predict.clone().detach()

    #         batch_perms_predict = np.zeros((batch_data.P.B, num_perm))
    #         batch_perms_diff = np.zeros((batch_data.P.B, num_perm))

    #         for i in range(num_perm) :
    #             batch_data.permute = True
    #             self.decoder(batch_data)

    #             predict = torch.argmax(batch_data.predict, dim=-1)
    #             batch_perms_predict[:, i] = predict.cpu().data.numpy()
            
    #             predict_difference = self.adversary_multi.output_diff(batch_data.predict, predict_true)
    #             batch_perms_diff[:, i] = predict_difference.squeeze(-1).cpu().data.numpy()
                
    #         permutations_predict.append(batch_perms_predict)
    #         permutations_diff.append(batch_perms_diff)

    #     permutations_predict = [x for y in permutations_predict for x in y]
    #     permutations_diff = [x for y in permutations_diff for x in y]
        
    #     return permutations_predict, permutations_diff

    # def adversarial_multi(self, data) :
    #     docs = data.P
    #     questions = data.Q
    #     entity_masks = data.E

    #     self.Pencoder.eval()
    #     self.Qencoder.eval()
    #     self.decoder.eval()

    #     print(self.adversary_multi.K)
        
    #     self.params = list(self.Pencoder.parameters()) + list(self.Qencoder.parameters()) + list(self.decoder.parameters())

    #     for p in self.params :
    #         p.requires_grad = False

    #     bsize = self.bsize
    #     N = len(questions)
    #     batches = list(range(0, N, bsize))

    #     outputs, attns, diffs = [], [], []

    #     for n in tqdm(batches) :
    #         torch.cuda.empty_cache()
    #         batch_doc = docs[n:n+bsize]
    #         batch_ques = questions[n:n+bsize]
    #         batch_entity_masks = entity_masks[n:n+bsize]

    #         batch_doc = BatchHolder(batch_doc)
    #         batch_ques = BatchHolder(batch_ques)

    #         batch_data = BatchMultiHolder(P=batch_doc, Q=batch_ques)
    #         batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

    #         self.Pencoder(batch_data.P)
    #         self.Qencoder(batch_data.Q)
    #         self.decoder(batch_data)

    #         self.adversary_multi(batch_data)

    #         predict_volatile = torch.argmax(batch_data.predict_volatile, dim=-1)
    #         outputs.append(predict_volatile.cpu().data.numpy())
            
    #         attn = batch_data.attn_volatile
    #         attns.append(attn.cpu().data.numpy())

    #         predict_difference = self.adversary_multi.output_diff(batch_data.predict_volatile, batch_data.predict.unsqueeze(1))
    #         diffs.append(predict_difference.cpu().data.numpy())

    #     outputs = [x for y in outputs for x in y]
    #     attns = [x for y in attns for x in y]
    #     diffs = [x for y in diffs for x in y]
        
    #     return outputs, attns, diffs

    # def gradient_mem(self, data) :
    #     docs = data.P
    #     questions = data.Q
    #     entity_masks = data.E

    #     self.Pencoder.train()
    #     self.Qencoder.train()
    #     self.decoder.train()
        
    #     bsize = self.bsize
    #     N = len(questions)

    #     grads = {'XxE' : [], 'XxE[X]' : [], 'H' : []}

    #     for n in range(0, N, bsize) :
    #         torch.cuda.empty_cache()
    #         batch_doc = docs[n:n+bsize]
    #         batch_ques = questions[n:n+bsize]
    #         batch_entity_masks = entity_masks[n:n+bsize]

    #         batch_doc = BatchHolder(batch_doc)
    #         batch_ques = BatchHolder(batch_ques)

    #         batch_data = BatchMultiHolder(P=batch_doc, Q=batch_ques)
    #         batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

    #         batch_data.P.keep_grads = True
    #         batch_data.detach = True

    #         self.Pencoder(batch_data.P)
    #         self.Qencoder(batch_data.Q)
    #         self.decoder(batch_data)
            
    #         max_predict = torch.argmax(batch_data.predict, dim=-1)
    #         prob_predict = nn.Softmax(dim=-1)(batch_data.predict)

    #         max_class_prob = torch.gather(prob_predict, -1, max_predict.unsqueeze(-1))
    #         max_class_prob.sum().backward()

    #         g = batch_data.P.embedding.grad
    #         em = batch_data.P.embedding
    #         g1 = (g * em).sum(-1)
            
    #         grads['XxE[X]'].append(g1.cpu().data.numpy())
            
    #         g1 = (g * self.Pencoder.embedding.weight.sum(0)).sum(-1)
    #         grads['XxE'].append(g1.cpu().data.numpy())
            
    #         g1 = batch_data.P.hidden.grad.sum(-1)
    #         grads['H'].append(g1.cpu().data.numpy())


    #     for k in grads :
    #         grads[k] = [x for y in grads[k] for x in y]
                    
    #     return grads       

    # def remove_and_run(self, data) :
    #     docs = data.P
    #     questions = data.Q
    #     entity_masks = data.E

    #     self.Pencoder.train()
    #     self.Qencoder.train()
    #     self.decoder.train()
        
    #     bsize = self.bsize
    #     N = len(questions)
    #     output_diffs = []

    #     for n in tqdm(range(0, N, bsize)) :
    #         torch.cuda.empty_cache()
    #         batch_doc = docs[n:n+bsize]
    #         batch_ques = questions[n:n+bsize]
    #         batch_entity_masks = entity_masks[n:n+bsize]

    #         batch_doc = BatchHolder(batch_doc)
    #         batch_ques = BatchHolder(batch_ques)

    #         batch_data = BatchMultiHolder(P=batch_doc, Q=batch_ques)
    #         batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

    #         self.Pencoder(batch_data.P)
    #         self.Qencoder(batch_data.Q)
    #         self.decoder(batch_data)

    #         po = np.zeros((batch_data.P.B, batch_data.P.maxlen))

    #         for i in range(1, batch_data.P.maxlen - 1) :
    #             batch_doc = BatchHolder(docs[n:n+bsize])

    #             batch_doc.seq = torch.cat([batch_doc.seq[:, :i], batch_doc.seq[:, i+1:]], dim=-1)
    #             batch_doc.lengths = batch_doc.lengths - 1
    #             batch_doc.masks = torch.cat([batch_doc.masks[:, :i], batch_doc.masks[:, i+1:]], dim=-1)

    #             batch_data_loop = BatchMultiHolder(P=batch_doc, Q=batch_ques)
    #             batch_data_loop.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

    #             self.Pencoder(batch_data_loop.P)
    #             self.decoder(batch_data_loop)

    #             predict_difference = self.adversary_multi.output_diff(batch_data_loop.predict, batch_data.predict)

    #             po[:, i] = predict_difference.squeeze(-1).cpu().data.numpy()

    #         output_diffs.append(po)

    #     output_diffs = [x for y in output_diffs for x in y]
        
    #     return output_diffs

