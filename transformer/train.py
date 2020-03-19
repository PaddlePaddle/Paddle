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

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable

from utils.configure import PDConfig
from utils.check import check_gpu, check_version

# include task-specific libs
import reader
from transformer import Transformer, CrossEntropyCriterion, NoamDecay
from model import Input


def do_train(args):
    trainer_count = 1  #get_nranks()

    @contextlib.contextmanager
    def null_guard():
        yield

    guard = fluid.dygraph.guard() if args.eager_run else null_guard()

    # define the data generator
    processor = reader.DataProcessor(
        fpattern=args.training_file,
        src_vocab_fpath=args.src_vocab_fpath,
        trg_vocab_fpath=args.trg_vocab_fpath,
        token_delimiter=args.token_delimiter,
        use_token_batch=args.use_token_batch,
        batch_size=args.batch_size,
        device_count=trainer_count,
        pool_size=args.pool_size,
        sort_type=args.sort_type,
        shuffle=args.shuffle,
        shuffle_batch=args.shuffle_batch,
        start_mark=args.special_token[0],
        end_mark=args.special_token[1],
        unk_mark=args.special_token[2],
        max_length=args.max_length,
        n_head=args.n_head)
    batch_generator = processor.data_generator(phase="train")
    if trainer_count > 1:  # for multi-process gpu training
        batch_generator = fluid.contrib.reader.distributed_batch_reader(
            batch_generator)
    if args.validation_file:
        val_processor = reader.DataProcessor(
            fpattern=args.validation_file,
            src_vocab_fpath=args.src_vocab_fpath,
            trg_vocab_fpath=args.trg_vocab_fpath,
            token_delimiter=args.token_delimiter,
            use_token_batch=args.use_token_batch,
            batch_size=args.batch_size,
            device_count=trainer_count,
            pool_size=args.pool_size,
            sort_type=args.sort_type,
            shuffle=False,
            shuffle_batch=False,
            start_mark=args.special_token[0],
            end_mark=args.special_token[1],
            unk_mark=args.special_token[2],
            max_length=args.max_length,
            n_head=args.n_head)
        val_batch_generator = val_processor.data_generator(phase="train")
    args.src_vocab_size, args.trg_vocab_size, args.bos_idx, args.eos_idx, \
        args.unk_idx = processor.get_vocab_summary()

    with guard:
        # set seed for CE
        random_seed = eval(str(args.random_seed))
        if random_seed is not None:
            fluid.default_main_program().random_seed = random_seed
            fluid.default_startup_program().random_seed = random_seed

        # define data loader
        train_loader = batch_generator
        if args.validation_file:
            val_loader = val_batch_generator

        # define model
        inputs = [
            Input(
                [None, None], "int64", name="src_word"), Input(
                    [None, None], "int64", name="src_pos"), Input(
                        [None, args.n_head, None, None],
                        "float32",
                        name="src_slf_attn_bias"), Input(
                            [None, None], "int64", name="trg_word"), Input(
                                [None, None], "int64", name="trg_pos"), Input(
                                    [None, args.n_head, None, None],
                                    "float32",
                                    name="trg_slf_attn_bias"), Input(
                                        [None, args.n_head, None, None],
                                        "float32",
                                        name="trg_src_attn_bias")
        ]
        labels = [
            Input(
                [None, 1], "int64", name="label"), Input(
                    [None, 1], "float32", name="weight")
        ]

        transformer = Transformer(
            args.src_vocab_size, args.trg_vocab_size, args.max_length + 1,
            args.n_layer, args.n_head, args.d_key, args.d_value, args.d_model,
            args.d_inner_hid, args.prepostprocess_dropout,
            args.attention_dropout, args.relu_dropout, args.preprocess_cmd,
            args.postprocess_cmd, args.weight_sharing, args.bos_idx,
            args.eos_idx)

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
                os.path.join(args.init_from_pretrain_model, "transformer"))

        # the best cross-entropy value with label smoothing
        loss_normalizer = -(
            (1. - args.label_smooth_eps) * np.log(
                (1. - args.label_smooth_eps)) + args.label_smooth_eps *
            np.log(args.label_smooth_eps / (args.trg_vocab_size - 1) + 1e-20))

        step_idx = 0
        # train loop
        for pass_id in range(args.epoch):
            pass_start_time = time.time()
            batch_id = 0
            for input_data in train_loader():
                losses = transformer.train(input_data[:-2], input_data[-2:])

                if step_idx % args.print_step == 0:
                    total_avg_cost = np.sum(losses)

                    if step_idx == 0:
                        logging.info(
                            "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                            "normalized loss: %f, ppl: %f" %
                            (step_idx, pass_id, batch_id, total_avg_cost,
                             total_avg_cost - loss_normalizer,
                             np.exp([min(total_avg_cost, 100)])))
                        avg_batch_time = time.time()
                    else:
                        logging.info(
                            "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                            "normalized loss: %f, ppl: %f, speed: %.2f step/s"
                            %
                            (step_idx, pass_id, batch_id, total_avg_cost,
                             total_avg_cost - loss_normalizer,
                             np.exp([min(total_avg_cost, 100)]),
                             args.print_step / (time.time() - avg_batch_time)))
                        avg_batch_time = time.time()

                if step_idx % args.save_step == 0 and step_idx != 0:
                    # validation: how to accumulate with Model loss
                    if args.validation_file:
                        total_avg_cost = 0
                        for idx, input_data in enumerate(val_loader()):
                            losses = transformer.eval(input_data[:-2],
                                                      input_data[-2:])
                            total_avg_cost += np.sum(losses)
                        total_avg_cost /= idx + 1
                        logging.info("validation, step_idx: %d, avg loss: %f, "
                                     "normalized loss: %f, ppl: %f" %
                                     (step_idx, total_avg_cost,
                                      total_avg_cost - loss_normalizer,
                                      np.exp([min(total_avg_cost, 100)])))

                    transformer.save(
                        os.path.join(args.save_model, "step_" + str(step_idx),
                                     "transformer"))

                batch_id += 1
                step_idx += 1

        time_consumed = time.time() - pass_start_time

        if args.save_model:
            transformer.save(
                os.path.join(args.save_model, "step_final", "transformer"))


if __name__ == "__main__":
    args = PDConfig(yaml_file="./transformer.yaml")
    args.build()
    args.Print()
    check_gpu(args.use_cuda)
    check_version()

    do_train(args)
