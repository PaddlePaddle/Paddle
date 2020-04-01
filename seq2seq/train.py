# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import os
import six
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import contextlib
from functools import partial

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.io import DataLoader
from paddle.fluid.dygraph_grad_clip import GradClipByGlobalNorm

import reader
from args import parse_args
from seq2seq_base import BaseModel, CrossEntropyCriterion
from seq2seq_attn import AttentionModel
from model import Input, set_device
from callbacks import ProgBarLogger
from metrics import Metric


class PPL(Metric):
    pass


def do_train(args):
    device = set_device("gpu" if args.use_gpu else "cpu")
    fluid.enable_dygraph(device)  #if args.eager_run else None

    # define model
    inputs = [
        Input(
            [None, None], "int64", name="src_word"),
        Input(
            [None], "int64", name="src_length"),
        Input(
            [None, None], "int64", name="trg_word"),
        Input(
            [None], "int64", name="trg_length"),
    ]
    labels = [Input([None, None, 1], "int64", name="label"), ]

    model = AttentionModel(args.src_vocab_size, args.tar_vocab_size,
                           args.hidden_size, args.hidden_size, args.num_layers,
                           args.dropout)

    model.prepare(
        fluid.optimizer.Adam(
            learning_rate=args.learning_rate,
            parameter_list=model.parameters()),
        CrossEntropyCriterion(),
        inputs=inputs,
        labels=labels)

    batch_size = 32
    src_seq_len = 10
    trg_seq_len = 12
    iter_num = 10

    def random_generator():
        for i in range(iter_num):
            src = np.random.randint(2, args.src_vocab_size,
                                    (batch_size, src_seq_len)).astype("int64")
            src_length = np.random.randint(1, src_seq_len,
                                           (batch_size, )).astype("int64")
            trg = np.random.randint(2, args.tar_vocab_size,
                                    (batch_size, trg_seq_len)).astype("int64")
            trg_length = np.random.randint(1, trg_seq_len,
                                           (batch_size, )).astype("int64")
            label = np.random.randint(
                1, trg_seq_len, (batch_size, trg_seq_len, 1)).astype("int64")
            yield src, src_length, trg, trg_length, label

    model.fit(train_data=random_generator, log_freq=1)
    exit(0)

    data_loaders = [None, None]
    data_files = [args.training_file, args.validation_file
                  ] if args.validation_file else [args.training_file]
    train_loader, eval_loader = data_loaders

    model.fit(train_data=train_loader,
              eval_data=None,
              epochs=1,
              eval_freq=1,
              save_freq=1,
              verbose=2)


if __name__ == "__main__":
    args = parse_args()
    do_train(args)
