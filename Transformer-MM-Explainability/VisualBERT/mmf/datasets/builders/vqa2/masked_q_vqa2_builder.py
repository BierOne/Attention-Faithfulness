# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import warnings

from VisualBERT.mmf.common.registry import registry
from VisualBERT.mmf.datasets.builders.vqa2.builder import VQA2Builder
from VisualBERT.mmf.datasets.builders.vqa2.masked_q_vqa2_dataset import MaskedQVQA2Dataset
from VisualBERT.mmf.datasets.concat_dataset import MMFConcatDataset


@registry.register_builder("masked_q_vqa2")
class MaskedQVQA2Builder(VQA2Builder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "masked_q_vqa2"
        self.dataset_class = MaskedQVQA2Dataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/vqa2/masked_q.yaml"
