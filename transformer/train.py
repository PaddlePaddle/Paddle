# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from utils.configure import PDConfig
from utils.check import check_gpu, check_version

# include task-specific libs
from reader import prepare_train_input, Seq2SeqDataset, Seq2SeqBatchSampler
from transformer import Transformer, CrossEntropyCriterion, NoamDecay
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
        Input([None, None], "int64", name="src_pos"),
        Input([None, args.n_head, None, None],
              "float32",
              name="src_slf_attn_bias"),
        Input([None, None], "int64", name="trg_word"),
        Input([None, None], "int64", name="trg_pos"),
        Input([None, args.n_head, None, None],
              "float32",
              name="trg_slf_attn_bias"),
        Input([None, args.n_head, None, None],
              "float32",
              name="trg_src_attn_bias"),
    ]
    labels = [
        Input(
            [None, 1], "int64", name="label"),
        Input(
            [None, 1], "float32", name="weight"),
    ]

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
                                                 trg_pad_idx=args.eos_idx,
                                                 n_head=args.n_head),
                              num_workers=0,
                              return_list=True)

    transformer = Transformer(
        args.src_vocab_size, args.trg_vocab_size, args.max_length + 1,
        args.n_layer, args.n_head, args.d_key, args.d_value, args.d_model,
        args.d_inner_hid, args.prepostprocess_dropout, args.attention_dropout,
        args.relu_dropout, args.preprocess_cmd, args.postprocess_cmd,
        args.weight_sharing, args.bos_idx, args.eos_idx)

    transformer.prepare(
        fluid.optimizer.Adam(
            learning_rate=fluid.layers.noam_decay(
                args.d_model, args.warmup_steps),  # args.learning_rate),
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=float(args.eps),
            parameter_list=transformer.parameters()),
        CrossEntropyCriterion(args.label_smooth_eps),
        inputs=inputs,
        labels=labels)

    ## init from some checkpoint, to resume the previous training
    if args.init_from_checkpoint:
        transformer.load(
            os.path.join(args.init_from_checkpoint, "transformer"))
    ## init from some pretrain models, to better solve the current task
    if args.init_from_pretrain_model:
        transformer.load(
            os.path.join(args.init_from_pretrain_model, "transformer"),
            reset_optimizer=True)

    # the best cross-entropy value with label smoothing
    loss_normalizer = -(
        (1. - args.label_smooth_eps) * np.log(
            (1. - args.label_smooth_eps)) + args.label_smooth_eps *
        np.log(args.label_smooth_eps / (args.trg_vocab_size - 1) + 1e-20))

    transformer.fit(train_data=train_loader,
                    eval_data=None,
                    epochs=1,
                    eval_freq=1,
                    save_freq=1,
                    verbose=2,
                    callbacks=[
                        LoggerCallback(
                            log_freq=args.print_step,
                            loss_normalizer=loss_normalizer)
                    ])


if __name__ == "__main__":
    args = PDConfig(yaml_file="./transformer.yaml")
    args.build()
    args.Print()
    check_gpu(args.use_cuda)
    check_version()

    do_train(args)
