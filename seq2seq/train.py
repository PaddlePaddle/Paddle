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
from callbacks import ProgBarLogger
from args import parse_args
from seq2seq_base import BaseModel, CrossEntropyCriterion
from seq2seq_attn import AttentionModel
from reader import Seq2SeqDataset, Seq2SeqBatchSampler, SortType, prepare_train_input


def do_train(args):
    device = set_device("gpu" if args.use_gpu else "cpu")
    fluid.enable_dygraph(device) if args.eager_run else None

    if args.enable_ce:
        fluid.default_main_program().random_seed = 102
        fluid.default_startup_program().random_seed = 102
        args.shuffle = False

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

    # def dataloader
    data_loaders = [None, None]
    data_prefixes = [args.train_data_prefix, args.eval_data_prefix
                     ] if args.eval_data_prefix else [args.train_data_prefix]
    for i, data_prefix in enumerate(data_prefixes):
        dataset = Seq2SeqDataset(
            fpattern=data_prefix + "." + args.src_lang,
            trg_fpattern=data_prefix + "." + args.tar_lang,
            src_vocab_fpath=args.vocab_prefix + "." + args.src_lang,
            trg_vocab_fpath=args.vocab_prefix + "." + args.tar_lang,
            token_delimiter=None,
            start_mark="<s>",
            end_mark="</s>",
            unk_mark="<unk>")
        (args.src_vocab_size, args.trg_vocab_size, bos_id, eos_id,
         unk_id) = dataset.get_vocab_summary()
        batch_sampler = Seq2SeqBatchSampler(
            dataset=dataset,
            use_token_batch=False,
            batch_size=args.batch_size,
            pool_size=args.batch_size * 20,
            sort_type=SortType.POOL,
            shuffle=args.shuffle)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            places=device,
            feed_list=None if fluid.in_dygraph_mode() else
            [x.forward() for x in inputs + labels],
            collate_fn=partial(
                prepare_train_input,
                bos_id=bos_id,
                eos_id=eos_id,
                pad_id=eos_id),
            num_workers=0,
            return_list=True)
        data_loaders[i] = data_loader
    train_loader, eval_loader = data_loaders

    model_maker = AttentionModel if args.attention else BaseModel
    model = model_maker(args.src_vocab_size, args.tar_vocab_size,
                        args.hidden_size, args.hidden_size, args.num_layers,
                        args.dropout)

    model.prepare(
        fluid.optimizer.Adam(
            learning_rate=args.learning_rate,
            parameter_list=model.parameters()),
        CrossEntropyCriterion(),
        inputs=inputs,
        labels=labels)
    model.fit(train_data=train_loader,
              eval_data=eval_loader,
              epochs=args.max_epoch,
              eval_freq=1,
              save_freq=1,
              save_dir=args.model_path,
              log_freq=1,
              verbose=2)


if __name__ == "__main__":
    args = parse_args()
    do_train(args)
