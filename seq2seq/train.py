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
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
from functools import partial

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.io import DataLoader

from model import Input, set_device
from metrics import Metric
from callbacks import ProgBarLogger
from args import parse_args
from seq2seq_base import BaseModel, CrossEntropyCriterion
from seq2seq_attn import AttentionModel
from reader import create_data_loader


class TrainCallback(ProgBarLogger):
    def __init__(self, args, ppl, verbose=2):
        super(TrainCallback, self).__init__(1, verbose)
        # control metric
        self.ppl = ppl
        self.batch_size = args.batch_size

    def on_train_begin(self, logs=None):
        super(TrainCallback, self).on_train_begin(logs)
        self.train_metrics += ["ppl"]  # remove loss to not print it
        self.ppl.reset()

    def on_train_batch_end(self, step, logs=None):
        batch_loss = logs["loss"][0]
        self.ppl.total_loss += batch_loss * self.batch_size
        logs["ppl"] = np.exp(self.ppl.total_loss / self.ppl.word_count)
        if step > 0 and step % self.ppl.reset_freq == 0:
            self.ppl.reset()
        super(TrainCallback, self).on_train_batch_end(step, logs)

    def on_eval_begin(self, logs=None):
        super(TrainCallback, self).on_eval_begin(logs)
        self.eval_metrics = ["ppl"]
        self.ppl.reset()

    def on_eval_batch_end(self, step, logs=None):
        batch_loss = logs["loss"][0]
        self.ppl.total_loss += batch_loss * self.batch_size
        logs["ppl"] = np.exp(self.ppl.total_loss / self.ppl.word_count)
        super(TrainCallback, self).on_eval_batch_end(step, logs)


class PPL(Metric):
    def __init__(self, reset_freq=100, name=None):
        super(PPL, self).__init__()
        self._name = name or "ppl"
        self.reset_freq = reset_freq
        self.reset()

    def add_metric_op(self, pred, label):
        seq_length = label[0]
        word_num = fluid.layers.reduce_sum(seq_length)
        return word_num

    def update(self, word_num):
        self.word_count += word_num
        return word_num

    def reset(self):
        self.total_loss = 0
        self.word_count = 0

    def accumulate(self):
        return self.word_count

    def name(self):
        return self._name


def do_train(args):
    device = set_device("gpu" if args.use_gpu else "cpu")
    fluid.enable_dygraph(device) if args.eager_run else None

    if args.enable_ce:
        fluid.default_main_program().random_seed = 102
        fluid.default_startup_program().random_seed = 102

    # define model
    inputs = [
        Input(
            [None, None], "int64", name="src_word"),
        Input(
            [None], "int64", name="src_length"),
        Input(
            [None, None], "int64", name="trg_word"),
    ]
    labels = [
        Input(
            [None], "int64", name="trg_length"),
        Input(
            [None, None, 1], "int64", name="label"),
    ]

    # def dataloader
    train_loader, eval_loader = create_data_loader(args, device)

    model_maker = AttentionModel if args.attention else BaseModel
    model = model_maker(args.src_vocab_size, args.tar_vocab_size,
                        args.hidden_size, args.hidden_size, args.num_layers,
                        args.dropout)
    optimizer = fluid.optimizer.Adam(
        learning_rate=args.learning_rate, parameter_list=model.parameters())
    optimizer._grad_clip = fluid.clip.GradientClipByGlobalNorm(
        clip_norm=args.max_grad_norm)
    ppl_metric = PPL()
    model.prepare(
        optimizer,
        CrossEntropyCriterion(),
        ppl_metric,
        inputs=inputs,
        labels=labels)
    model.fit(train_data=train_loader,
              eval_data=eval_loader,
              epochs=args.max_epoch,
              eval_freq=1,
              save_freq=1,
              save_dir=args.model_path,
              callbacks=[TrainCallback(args, ppl_metric)])


if __name__ == "__main__":
    args = parse_args()
    do_train(args)
