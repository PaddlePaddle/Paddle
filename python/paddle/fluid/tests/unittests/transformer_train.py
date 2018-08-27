# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import ast
import multiprocessing
import os
import time
from functools import partial

import numpy as np
import paddle.fluid as fluid

import transformer_reader as reader
from transformer_config import *
from transformer_model import transformer, position_encoding_init
from optim import LearningRateScheduler


def pad_batch_data(insts,
                   pad_idx,
                   n_head,
                   is_target=False,
                   is_label=False,
                   return_attn_bias=True,
                   return_max_len=True,
                   return_num_token=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    num_token = reduce(lambda x, y: x + y,
                       [len(inst) for inst in insts]) if return_num_token else 0
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.
    inst_data = np.array(
        [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, 1])]
    if is_label:  # label weight
        inst_weight = np.array(
            [[1.] * len(inst) + [0.] * (max_len - len(inst)) for inst in insts])
        return_list += [inst_weight.astype("float32").reshape([-1, 1])]
    else:  # position data
        inst_pos = np.array([
            range(1, len(inst) + 1) + [0] * (max_len - len(inst))
            for inst in insts
        ])
        return_list += [inst_pos.astype("int64").reshape([-1, 1])]
    if return_attn_bias:
        if is_target:
            # This is used to avoid attention on paddings and subsequent
            # words.
            slf_attn_bias_data = np.ones((inst_data.shape[0], max_len, max_len))
            slf_attn_bias_data = np.triu(slf_attn_bias_data,
                                         1).reshape([-1, 1, max_len, max_len])
            slf_attn_bias_data = np.tile(slf_attn_bias_data,
                                         [1, n_head, 1, 1]) * [-1e9]
        else:
            # This is used to avoid attention on paddings.
            slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                           (max_len - len(inst))
                                           for inst in insts])
            slf_attn_bias_data = np.tile(
                slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                [1, n_head, max_len, 1])
        return_list += [slf_attn_bias_data.astype("float32")]
    if return_max_len:
        return_list += [max_len]
    if return_num_token:
        return_list += [num_token]
    return return_list if len(return_list) > 1 else return_list[0]


def prepare_batch_input(insts, data_input_names, src_pad_idx, trg_pad_idx,
                        n_head, d_model):
    """
    Put all padded data needed by training into a dict.
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    src_word = src_word.reshape(-1, src_max_len, 1)
    src_pos = src_pos.reshape(-1, src_max_len, 1)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, n_head, is_target=True)
    trg_word = trg_word.reshape(-1, trg_max_len, 1)
    trg_pos = trg_pos.reshape(-1, trg_max_len, 1)

    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, trg_max_len, 1]).astype("float32")

    lbl_word, lbl_weight, num_token = pad_batch_data(
        [inst[2] for inst in insts],
        trg_pad_idx,
        n_head,
        is_target=False,
        is_label=True,
        return_attn_bias=False,
        return_max_len=False,
        return_num_token=True)

    data_input_dict = dict(
        zip(data_input_names, [
            src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos,
            trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight
        ]))
    return data_input_dict, np.asarray([num_token], dtype="float32")


def read_multiple(reader, count, clip_last=True):
    """
    Stack data from reader for multi-devices.
    """

    def __impl__():
        res = []
        for item in reader():
            res.append(item)
            if len(res) == count:
                yield res
                res = []
        if len(res) == count:
            yield res
        elif not clip_last:
            data = []
            for item in res:
                data += item
            if len(data) > count:
                inst_num_per_part = len(data) // count
                yield [
                    data[inst_num_per_part * i:inst_num_per_part * (i + 1)]
                    for i in range(count)
                ]

    return __impl__


def split_data(data, num_part):
    """
    Split data for each device.
    """
    if len(data) == num_part:
        return data
    data = data[0]
    inst_num_per_part = len(data) // num_part
    return [
        data[inst_num_per_part * i:inst_num_per_part * (i + 1)]
        for i in range(num_part)
    ]


def test_context(train_progm, avg_cost, train_exe, dev_count, data_input_names,
                 sum_cost, token_num):
    # Context to do validation.
    test_program = train_progm.clone()
    with fluid.program_guard(test_program):
        test_program = fluid.io.get_inference_program([avg_cost])

    val_data = reader.DataReader(
        src_vocab_fpath=TrainTaskConfig.src_vocab_fpath,
        trg_vocab_fpath=TrainTaskConfig.trg_vocab_fpath,
        fpattern=TrainTaskConfig.val_file_pattern,
        token_delimiter=TrainTaskConfig.token_delimiter,
        use_token_batch=TrainTaskConfig.use_token_batch,
        batch_size=TrainTaskConfig.batch_size *
        (1 if TrainTaskConfig.use_token_batch else dev_count),
        pool_size=TrainTaskConfig.pool_size,
        sort_type=TrainTaskConfig.sort_type,
        start_mark=TrainTaskConfig.special_token[0],
        end_mark=TrainTaskConfig.special_token[1],
        unk_mark=TrainTaskConfig.special_token[2],
        # count start and end tokens out
        max_length=ModelHyperParams.max_length - 2,
        clip_last_batch=False,
        shuffle=False,
        shuffle_batch=False)

    build_strategy = fluid.BuildStrategy()

    strategy = fluid.ExecutionStrategy()
    strategy.num_threads = 1

    test_exe = fluid.ParallelExecutor(
        use_cuda=TrainTaskConfig.use_gpu,
        main_program=test_program,
        share_vars_from=train_exe,
        build_strategy=build_strategy)

    def test(exe=test_exe):
        test_total_cost = 0
        test_total_token = 0
        test_data = read_multiple(
            reader=val_data.batch_generator,
            count=dev_count if TrainTaskConfig.use_token_batch else 1)
        for batch_id, data in enumerate(test_data()):
            feed_list = []
            for place_id, data_buffer in enumerate(
                    split_data(
                        data, num_part=dev_count)):
                data_input_dict, _ = prepare_batch_input(
                    data_buffer, data_input_names, ModelHyperParams.eos_idx,
                    ModelHyperParams.eos_idx, ModelHyperParams.n_head,
                    ModelHyperParams.d_model)
                feed_list.append(data_input_dict)

            outs = exe.run(feed=feed_list,
                           fetch_list=[sum_cost.name, token_num.name])
            sum_cost_val, token_num_val = np.array(outs[0]), np.array(outs[1])
            test_total_cost += sum_cost_val.sum()
            test_total_token += token_num_val.sum()
        test_avg_cost = test_total_cost / test_total_token
        test_ppl = np.exp([min(test_avg_cost, 100)])
        return test_avg_cost, test_ppl

    return test


def train_loop(exe, train_progm, dev_count, sum_cost, avg_cost, lr_scheduler,
               token_num, predict):
    # Initialize the parameters.
    if TrainTaskConfig.ckpt_path:
        lr_scheduler.current_steps = TrainTaskConfig.start_step
    else:
        exe.run(fluid.framework.default_startup_program())

    train_data = reader.DataReader(
        src_vocab_fpath=TrainTaskConfig.src_vocab_fpath,
        trg_vocab_fpath=TrainTaskConfig.trg_vocab_fpath,
        fpattern=TrainTaskConfig.train_file_pattern,
        token_delimiter=TrainTaskConfig.token_delimiter,
        use_token_batch=TrainTaskConfig.use_token_batch,
        batch_size=TrainTaskConfig.batch_size *
        (1 if TrainTaskConfig.use_token_batch else dev_count),
        pool_size=TrainTaskConfig.pool_size,
        sort_type=TrainTaskConfig.sort_type,
        shuffle=TrainTaskConfig.shuffle,
        shuffle_batch=TrainTaskConfig.shuffle_batch,
        start_mark=TrainTaskConfig.special_token[0],
        end_mark=TrainTaskConfig.special_token[1],
        unk_mark=TrainTaskConfig.special_token[2],
        # count start and end tokens out
        max_length=ModelHyperParams.max_length - 2,
        clip_last_batch=False)
    train_data = read_multiple(
        reader=train_data.batch_generator,
        count=dev_count if TrainTaskConfig.use_token_batch else 1)

    build_strategy = fluid.BuildStrategy()
    # Since the token number differs among devices, customize gradient scale to
    # use token average cost among multi-devices. and the gradient scale is
    # `1 / token_number` for average cost.
    build_strategy.gradient_scale_strategy = fluid.BuildStrategy.GradientScaleStrategy.Customized

    strategy = fluid.ExecutionStrategy()
    strategy.num_threads = 1

    train_exe = fluid.ParallelExecutor(
        use_cuda=TrainTaskConfig.use_gpu,
        loss_name=sum_cost.name,
        main_program=train_progm,
        build_strategy=build_strategy,
        exec_strategy=strategy)

    data_input_names = encoder_data_input_fields + decoder_data_input_fields[:
                                                                             -1] + label_data_input_fields

    if TrainTaskConfig.val_file_pattern is not None:
        test = test_context(train_progm, avg_cost, train_exe, dev_count,
                            data_input_names, sum_cost, token_num)

    # the best cross-entropy value with label smoothing
    loss_normalizer = -((1. - TrainTaskConfig.label_smooth_eps) * np.log(
        (1. - TrainTaskConfig.label_smooth_eps
         )) + TrainTaskConfig.label_smooth_eps *
                        np.log(TrainTaskConfig.label_smooth_eps / (
                            ModelHyperParams.trg_vocab_size - 1) + 1e-20))
    init = False
    for pass_id in xrange(TrainTaskConfig.pass_num):
        pass_start_time = time.time()
        for batch_id, data in enumerate(train_data()):
            if batch_id >= 5:
                break

            feed_list = []
            total_num_token = 0
            if TrainTaskConfig.local:
                lr_rate = lr_scheduler.update_learning_rate()
            for place_id, data_buffer in enumerate(
                    split_data(
                        data, num_part=dev_count)):
                data_input_dict, num_token = prepare_batch_input(
                    data_buffer, data_input_names, ModelHyperParams.eos_idx,
                    ModelHyperParams.eos_idx, ModelHyperParams.n_head,
                    ModelHyperParams.d_model)
                total_num_token += num_token
                feed_kv_pairs = data_input_dict.items()
                if TrainTaskConfig.local:
                    feed_kv_pairs += {
                        lr_scheduler.learning_rate.name: lr_rate
                    }.items()
                feed_list.append(dict(feed_kv_pairs))

                if not init:
                    for pos_enc_param_name in pos_enc_param_names:
                        pos_enc = position_encoding_init(
                            ModelHyperParams.max_length + 1,
                            ModelHyperParams.d_model)
                        feed_list[place_id][pos_enc_param_name] = pos_enc
            for feed_dict in feed_list:
                feed_dict[sum_cost.name + "@GRAD"] = 1. / total_num_token

            outs = train_exe.run(fetch_list=[sum_cost.name, token_num.name],
                                 feed=feed_list)

            sum_cost_val, token_num_val = np.array(outs[0]), np.array(outs[1])
            total_sum_cost = sum_cost_val.sum()
            total_token_num = token_num_val.sum()
            total_avg_cost = total_sum_cost / total_token_num

            init = True

            # Validate and save the model for inference.
            if TrainTaskConfig.val_file_pattern is not None:
                val_avg_cost, val_ppl = test()
                print("[%f]" % val_avg_cost)
            else:
                assert (False)
