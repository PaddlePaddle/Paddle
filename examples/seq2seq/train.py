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
import random
from functools import partial

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.io import DataLoader

from hapi.model import Input, set_device
from args import parse_args
from seq2seq_base import BaseModel, CrossEntropyCriterion
from seq2seq_attn import AttentionModel
from reader import create_data_loader
from utility import PPL, TrainCallback


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
    grad_clip = fluid.clip.GradientClipByGlobalNorm(
        clip_norm=args.max_grad_norm)
    optimizer = fluid.optimizer.Adam(
        learning_rate=args.learning_rate,
        parameter_list=model.parameters(),
        grad_clip=grad_clip)

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
              callbacks=[TrainCallback(ppl_metric, args.log_freq)])


if __name__ == "__main__":
    args = parse_args()
    do_train(args)
