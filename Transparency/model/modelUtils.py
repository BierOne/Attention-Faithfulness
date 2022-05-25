def isTrue(obj, attr) :
    return hasattr(obj, attr) and getattr(obj, attr)

import numpy as np
import torch
from math import ceil


Paddings = [0, 2, 3] # MASK, Begin, End
def masked_softmax(tensor, mask, dim=-1) :
    # tensor : (x1, x2, x3, ..., xn) Tensor
    # mask : (x1, x2, x3, ..., xn) LongTensor containing 1/0 
    #        where 1 if element to be masked else 0
    # dim : dimension over which to do softmax
    tensor.masked_fill_(mask.long().bool(), -float('inf'))
    return torch.nn.Softmax(dim=dim)(tensor)

def get_sorting_index_with_noise_from_lengths(lengths, noise_frac) :
    if noise_frac > 0 :
        noisy_lengths = [x + np.random.randint(np.floor(-x*noise_frac), np.ceil(x*noise_frac)) for x in lengths]
    else :
        noisy_lengths = lengths
    return np.argsort(noisy_lengths)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class BatchHolder() : 
    def __init__(self, data, mask_ratio=None, exp_socre=None, exp_socre_ind=None, delta_type="slice_out") :
        maxlen = max([len(x) for x in data])
        self.maxlen = maxlen
        self.B = len(data)

        lengths = []
        expanded = []
        masks = []

        if mask_ratio:
            mask_weights = torch.zeros(self.B)
            for i, d in enumerate(data):
                len_d = len(d)
                num_masks = ceil((len_d-2) * mask_ratio)
                num_masked = 0

                d_masks = [1] + [0]*(len_d-2) + [1]
                for _weight, _ind in zip(exp_socre[i], exp_socre_ind[i]):
                    if d[_ind] in Paddings:
                        continue
                    if delta_type == "slice_out":
                        d.pop(_ind)
                        d_masks.pop(_ind)
                        len_d -= 1 # only slice_out minus 1
                    elif delta_type == "att_mask":
                        d_masks[_ind] = 1
                    elif delta_type == "zeros_mask":
                        d[_ind] = 0
                        d_masks[_ind] = 1
                    num_masked += 1
                    mask_weights[i] += _weight
                    if num_masked >= num_masks:
                        break

                rem = maxlen - len_d
                expanded.append(d + [0]*rem)
                lengths.append(len_d)
                masks.append(d_masks + [1]*rem)

            self.mask_weights = mask_weights
        else:
            for i, d in enumerate(data) :
                rem = maxlen - len(d)
                expanded.append(d + [0]*rem)
                lengths.append(len(d))
                masks.append([1] + [0]*(len(d)-2) + [1]*(rem+1))

        self.lengths = torch.LongTensor(np.array(lengths))
        self.seq = torch.LongTensor(np.array(expanded, dtype='int64')).to(device)

        self.masks = torch.ByteTensor(np.array(masks)).to(device) # 1 -> mask, 0 -> non-mask

        self.hidden = None
        self.predict = None
        self.attn = None
        self.after_mask = None


    def generate_permutation(self) :
        perm_idx = np.tile(np.arange(self.maxlen), (self.B, 1))

        for i, x in enumerate(self.lengths) :
            perm = np.random.permutation(x.item()-2) + 1
            perm_idx[i, 1:x-1] = perm

        return perm_idx

    def generate_uniform_attn(self) :
        attn = np.zeros((self.B, self.maxlen))
        inv_l = 1. / self.lengths.cpu().data.numpy()
        attn += inv_l[:, None]
        return torch.Tensor(attn).to(device)

class BatchMultiHolder() :
    def __init__(self, **holders) :
        for name, value in holders.items() :
            setattr(self, name, value)

def kld(a1, a2) :
    #(B, *, A), #(B, *, A)
    a1 = torch.clamp(a1, 0, 1)
    a2 = torch.clamp(a2, 0, 1)
    log_a1 = torch.log(a1 + 1e-10)
    log_a2 = torch.log(a2 + 1e-10)

    kld = a1 * (log_a1 - log_a2)
    kld = kld.sum(-1)

    return kld

def jsd(p, q) :
    m = 0.5 * (p + q)
    jsd = 0.5 * (kld(p, m) + kld(q, m))
    
    return jsd.unsqueeze(-1)
