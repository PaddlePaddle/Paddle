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
SequenceTagging network structure
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
from utils.configure import PDConfig
from utils.check import check_gpu, check_version
from utils.metrics import chunk_count
from reader import LacDataset, create_lexnet_data_generator, create_dataloader

work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(work_dir, "../"))
from hapi.model import set_device, Input

import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.layers.utils import flatten


def main(args):
    place = set_device(args.device)
    fluid.enable_dygraph(place) if args.dynamic else None

    inputs = [Input([None, None], 'int64', name='words'), 
              Input([None], 'int64', name='length')] 

    feed_list = None if args.dynamic else [x.forward() for x in inputs]
    dataset = LacDataset(args)
    eval_path = args.test_file

    chunk_evaluator = fluid.metrics.ChunkEvaluator()
    chunk_evaluator.reset()

    eval_generator = create_lexnet_data_generator(
        args, reader=dataset, file_name=eval_path, place=place, mode="test")

    eval_dataset = create_dataloader(
        eval_generator, place, feed_list=feed_list)

    vocab_size = dataset.vocab_size
    num_labels = dataset.num_labels
    model = SeqTagging(args, vocab_size, num_labels)

    optim = AdamOptimizer(
        learning_rate=args.base_learning_rate,
        parameter_list=model.parameters())

    model.mode = "test"
    model.prepare(inputs=inputs)
    model.load(args.init_from_checkpoint, skip_mismatch=True)

    for data in eval_dataset():
        if len(data) == 1: 
            batch_data = data[0]
            targets = np.array(batch_data[2])
        else: 
            batch_data = data
            targets = batch_data[2].numpy()
        inputs_data = [batch_data[0], batch_data[1]]
        crf_decode, length = model.test(inputs=inputs_data)
        num_infer_chunks, num_label_chunks, num_correct_chunks = chunk_count(crf_decode, targets, length, dataset.id2label_dict)
        chunk_evaluator.update(num_infer_chunks, num_label_chunks, num_correct_chunks)
    
    precision, recall, f1 = chunk_evaluator.eval()
    print("[test] P: %.5f, R: %.5f, F1: %.5f" % (precision, recall, f1))


if __name__ == '__main__': 
    args = PDConfig(yaml_file="sequence_tagging.yaml")
    args.build()
    args.Print()

    use_gpu = True if args.device == "gpu" else False
    check_gpu(use_gpu)
    check_version()
    main(args)
