# Copyright (c) Facebook, Inc. and its affiliates.
import argparse

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_core_args()

    def get_parser(self):
        return self.parser

    def add_core_args(self):
        self.parser.add_argument_group("Core Arguments")
        # TODO: Add Help flag here describing MMF Configuration
        # and point to configuration documentation
        self.parser.add_argument(
            "-co",
            "--config_override",
            type=str,
            default=None,
            help="Use to override config from command line directly",
        )
        # This is needed to support torch.distributed.launch
        self.parser.add_argument(
            "--local_rank", type=int, default=None, help="Local rank of the argument"
        )
        # perturbation configuration
        self.parser.add_argument('--method', type=str,
                            default='ours_no_lrp',
                            choices=["ours_no_lrp",
                                     "transformer_attribution",
                                    "partial_lrp",
                                    "raw_attn",
                                    "attn_gradcam",
                                    "rollout",
                                     "attn_grad",
                                     "attn_norm",
                                     "inputGrad",
                                     "ig", "rand0", "rand1", "rand2"],
                            help='')
        self.parser.add_argument('--num-samples', type=int,
                            default=10000,
                            help='')
        self.parser.add_argument('--is-positive-pert', type=boolean_string,
                            default=False,
                            help='')
        self.parser.add_argument('--is-text-pert', type=boolean_string,
                            default=False,
                            help='')
        self.parser.add_argument('--load_cached_exp', type=boolean_string,
                            default=False,
                            help='')
        self.parser.add_argument('--only_perturbation', type=boolean_string,
                            default=True,
                            help='')
        self.parser.add_argument('--exp_metric', type=str,
                            default="Violation",
                            choices=["AUCTP", "Sufficiency", "Comprehensiveness", "Violation", "RC"],
                            help='')
        self.parser.add_argument('--mask_size', type=int,
                            default=1,
                            help='')
        self.parser.add_argument('--delta_type', type=str,
                            default="att_mask",
                            choices=["zeros_mask", "slice_out", "att_mask"],
                            help='')
        self.parser.add_argument('--b_size', type=int,
                            default=1,
                            help='')
        self.parser.add_argument('--task', type=str,
                            default="vqa",
                            choices=["vqa", "gqa", "nlvr"],
                            help='')
        self.parser.add_argument('--save_comp', type=boolean_string,
                            default=False,
                            help='')
        self.parser.add_argument(
            "opts",
            default=None,
            nargs=argparse.REMAINDER,
            help="Modify config options from command line",
        )



flags = Flags()
