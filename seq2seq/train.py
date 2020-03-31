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

from configure import PDConfig
from reader import prepare_train_input, Seq2SeqDataset, Seq2SeqBatchSampler
from seq2seq import Seq2Seq, CrossEntropyCriterion
from model import Input, set_device
from callbacks import ProgBarLogger


class LoggerCallback(ProgBarLogger):
    def __init__(self, log_freq=1, verbose=2, loss_normalizer=0.):
        super(LoggerCallback, self).__init__(log_freq, verbose)
        # TODO: wrap these override function to simplify
        self.loss_normalizer = loss_normalizer

    def on_train_begin(self, logs=None):
        super(LoggerCallback, self).on_train_begin(logs)
        self.train_metrics += ["normalized loss", "ppl"]

    def on_train_batch_end(self, step, logs=None):
        logs["normalized loss"] = logs["loss"][0] - self.loss_normalizer
        logs["ppl"] = np.exp(min(logs["loss"][0], 100))
        super(LoggerCallback, self).on_train_batch_end(step, logs)

    def on_eval_begin(self, logs=None):
        super(LoggerCallback, self).on_eval_begin(logs)
        self.eval_metrics += ["normalized loss", "ppl"]

    def on_eval_batch_end(self, step, logs=None):
        logs["normalized loss"] = logs["loss"][0] - self.loss_normalizer
        logs["ppl"] = np.exp(min(logs["loss"][0], 100))
        super(LoggerCallback, self).on_eval_batch_end(step, logs)


def do_train(args):
    device = set_device("gpu" if args.use_cuda else "cpu")
    fluid.enable_dygraph(device) if args.eager_run else None

    # set seed for CE
    random_seed = eval(str(args.random_seed))
    if random_seed is not None:
        fluid.default_main_program().random_seed = random_seed
        fluid.default_startup_program().random_seed = random_seed

    # define model
    inputs = [
        Input([None, None], "int64", name="src_word"),
        Input([None], "int64", name="src_length"),
        Input([None, None], "int64", name="trg_word"),
        Input([None], "int64", name="trg_length"),
    ]
    labels = [
        Input([None, None, 1], "int64", name="label"),
    ]

    model = Seq2Seq(args.src_vocab_size, args.trg_vocab_size, args.embed_dim,
                    args.hidden_size, args.num_layers, args.dropout)

    model.prepare(fluid.optimizer.Adam(learning_rate=args.learning_rate,
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
            src_length = np.random.randint(
                1, src_seq_len, (batch_size, )).astype("int64")
            trg = np.random.randint(2, args.trg_vocab_size,
                                    (batch_size, trg_seq_len)).astype("int64")
            trg_length = np.random.randint(1, trg_seq_len,
                                        (batch_size, )).astype("int64")
            label = np.random.randint(1, trg_seq_len,
                                    (batch_size, trg_seq_len, 1)).astype("int64")
            yield src, src_length, trg, trg_length, label

    model.fit(train_data=random_generator, log_freq=1)
    exit(0)


    dataset = Seq2SeqDataset(fpattern=args.training_file,
                             src_vocab_fpath=args.src_vocab_fpath,
                             trg_vocab_fpath=args.trg_vocab_fpath,
                             token_delimiter=args.token_delimiter,
                             start_mark=args.special_token[0],
                             end_mark=args.special_token[1],
                             unk_mark=args.special_token[2])
    args.src_vocab_size, args.trg_vocab_size, args.bos_idx, args.eos_idx, \
        args.unk_idx = dataset.get_vocab_summary()
    batch_sampler = Seq2SeqBatchSampler(dataset=dataset,
                                        use_token_batch=args.use_token_batch,
                                        batch_size=args.batch_size,
                                        pool_size=args.pool_size,
                                        sort_type=args.sort_type,
                                        shuffle=args.shuffle,
                                        shuffle_batch=args.shuffle_batch,
                                        max_length=args.max_length)
    train_loader = DataLoader(dataset=dataset,
                              batch_sampler=batch_sampler,
                              places=device,
                              feed_list=[x.forward() for x in inputs + labels],
                              collate_fn=partial(prepare_train_input,
                                                 src_pad_idx=args.eos_idx,
                                                 trg_pad_idx=args.eos_idx),
                              num_workers=0,
                              return_list=True)

    model.fit(train_data=train_loader,
              eval_data=None,
              epochs=1,
              eval_freq=1,
              save_freq=1,
              verbose=2,
              callbacks=[
                  LoggerCallback(log_freq=args.print_step)
              ])


if __name__ == "__main__":
    args = PDConfig(yaml_file="./seq2seq.yaml")
    args.build()
    args.Print()

    do_train(args)
