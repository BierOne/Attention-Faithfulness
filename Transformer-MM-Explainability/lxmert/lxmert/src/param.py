# coding=utf-8
# Copyleft 2019 project LXRT.

import argparse
import random

import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'      # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def parse_args():
    parser = argparse.ArgumentParser()

    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", default=None)

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=256)
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Debugging
    parser.add_argument('--output', type=str, default='snap/test')
    parser.add_argument("--fast", action='store_const', default=False, const=True)
    parser.add_argument("--tiny", action='store_const', default=False, const=True)
    parser.add_argument("--tqdm", action='store_const', default=False, const=True)

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str, default=None,
                        help='Load the pre-trained LXMERT model.')
    parser.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str, default=None,
                        help='Load the pre-trained LXMERT model with QA answer head.')
    parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                        help='If none of the --load, --loadLXMERT, --loadLXMERTQA is set, '
                             'the model would be trained from scratch. If --fromScratch is'
                             ' not specified, the model would load BERT-pre-trained weights by'
                             ' default. ')

    # Optimization
    parser.add_argument("--mceLoss", dest='mce_loss', action='store_const', default=False, const=True)

    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument("--llayers", default=9, type=int, help='Number of Language layers')
    parser.add_argument("--xlayers", default=5, type=int, help='Number of CROSS-modality layers.')
    parser.add_argument("--rlayers", default=5, type=int, help='Number of object Relationship layers.')

    # LXMERT Pre-training Config
    parser.add_argument("--taskMatched", dest='task_matched', action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskLM", dest='task_mask_lm', action='store_const', default=False, const=True)
    parser.add_argument("--taskObjPredict", dest='task_obj_predict', action='store_const', default=False, const=True)
    parser.add_argument("--taskQA", dest='task_qa', action='store_const', default=False, const=True)
    parser.add_argument("--visualLosses", dest='visual_losses', default='obj,attr,feat', type=str)
    parser.add_argument("--qaSets", dest='qa_sets', default=None, type=str)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate', default=0.15, type=float)

    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=0)


    # perturbation configuration
    parser.add_argument('--method', type=str,
                        default='ours_no_lrp',
                        choices=['ours_with_lrp', 'rollout', 'partial_lrp', 'transformer_att',
                                 'raw_attn', 'attn_gradcam', 'ours_with_lrp_no_normalization', 'ours_no_lrp',
                                 'ours_no_lrp_no_norm', 'ablation_no_aggregation', 'ablation_no_self_in_10',
                                 'attn_norm', "rand0", "rand1", "rand2",
                                 'ig', 'inputGrad', 'attn_grad'],
                        help='')
    parser.add_argument('--num-samples', type=int,
                        default=10000,
                        help='')
    parser.add_argument('--is-positive-pert', type=boolean_string,
                        default=False,
                        help='')
    parser.add_argument('--is-text-pert', type=boolean_string,
                        default=False,
                        help='')
    parser.add_argument('--COCO_path', type=str,
                        default='',
                        help='path to COCO 2014 validation set')

    parser.add_argument('--load_cached_exp', type=boolean_string,
                             default=False,
                             help='')
    parser.add_argument('--load_raw_img', type=boolean_string,
                             default=False,
                             help='')
    parser.add_argument('--only_perturbation', type=boolean_string,
                             default=True,
                             help='')
    parser.add_argument('--exp_metric', type=str,
                             default="Violation",
                             choices=["all", "AUCTP", "Sufficiency", "Comprehensiveness", "Violation", "RC"],
                             help='')
    parser.add_argument('--mask_size', type=int,
                             default=1,
                             help='')
    parser.add_argument('--delta_type', type=str,
                             default="att_mask",
                             choices=["all", "zeros_mask", "slice_out", "att_mask"],
                             help='')

    parser.add_argument('--task', type=str,
                             default="vqa",
                             choices=["vqa", "gqa", "nlvr"],
                             help='')

    # Parse the arguments.
    args = parser.parse_args()

    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()
