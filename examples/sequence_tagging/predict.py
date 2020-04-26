#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SequenceTagging predict structure
"""

from __future__ import division
from __future__ import print_function

import io
import os
import sys
import math
import argparse
import numpy as np

from train import SeqTagging
from utils.check import check_gpu, check_version
from utils.configure import PDConfig
from reader import LacDataset, LacDataLoader

work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(work_dir, "../"))
from hapi.model import set_device, Input

import paddle.fluid as fluid
from paddle.fluid.layers.utils import flatten


def main(args):
    place = set_device(args.device)
    fluid.enable_dygraph(place) if args.dynamic else None

    inputs = [
        Input(
            [None, None], 'int64', name='words'), Input(
                [None], 'int64', name='length')
    ]

    dataset = LacDataset(args)
    predict_dataset = LacDataLoader(args, place, phase="predict")

    vocab_size = dataset.vocab_size
    num_labels = dataset.num_labels
    model = SeqTagging(args, vocab_size, num_labels, mode="predict")

    model.mode = "test"
    model.prepare(inputs=inputs)

    model.load(args.init_from_checkpoint, skip_mismatch=True)

    f = open(args.output_file, "wb")
    for data in predict_dataset.dataloader:
        if len(data) == 1:
            input_data = data[0]
        else:
            input_data = data
        results, length = model.test_batch(inputs=flatten(input_data))
        for i in range(len(results)):
            word_len = length[i]
            word_ids = results[i][:word_len]
            tags = [dataset.id2label_dict[str(id)] for id in word_ids]
            f.write("\002".join(tags) + "\n")


if __name__ == '__main__':
    args = PDConfig(yaml_file="sequence_tagging.yaml")
    args.build()
    args.Print()

    use_gpu = True if args.device == "gpu" else False
    check_gpu(use_gpu)
    check_version()
    main(args)
