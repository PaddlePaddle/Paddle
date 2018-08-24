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

import reader
from config import *
from model import transformer, position_encoding_init
from optim import LearningRateScheduler


def parse_args():
    parser = argparse.ArgumentParser("Training for Transformer.")
    parser.add_argument(
        "--src_vocab_fpath",
        type=str,
        required=True,
        help="The path of vocabulary file of source language.")
    parser.add_argument(
        "--trg_vocab_fpath",
        type=str,
        required=True,
        help="The path of vocabulary file of target language.")
    parser.add_argument(
        "--train_file_pattern",
        type=str,
        required=True,
        help="The pattern to match training data files.")
    parser.add_argument(
        "--val_file_pattern",
        type=str,
        help="The pattern to match validation data files.")
    parser.add_argument(
        "--use_token_batch",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to "
        "produce batch data according to token number.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="The number of sequences contained in a mini-batch, or the maximum "
        "number of tokens (include paddings) contained in a mini-batch. Note "
        "that this represents the number on single device and the actual batch "
        "size for multi-devices will multiply the device number.")
    parser.add_argument(
        "--pool_size",
        type=int,
        default=10000,
        help="The buffer size to pool data.")
    parser.add_argument(
        "--sort_type",
        default="pool",
        choices=("global", "pool", "none"),
        help="The grain to sort by length: global for all instances; pool for "
        "instances in pool; none for no sort.")
    parser.add_argument(
        "--shuffle",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to shuffle instances in each pass.")
    parser.add_argument(
        "--shuffle_batch",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to shuffle the data batches.")
    parser.add_argument(
        "--special_token",
        type=str,
        default=["<s>", "<e>", "<unk>"],
        nargs=3,
        help="The <bos>, <eos> and <unk> tokens in the dictionary.")
    parser.add_argument(
        "--token_delimiter",
        type=partial(
            str.decode, encoding="string-escape"),
        default=" ",
        help="The delimiter used to split tokens in source or target sentences. "
        "For EN-DE BPE data we provided, use spaces as token delimiter. "
        "For EN-FR wordpiece data we provided, use '\x01' as token delimiter.")
    parser.add_argument(
        'opts',
        help='See config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    parser.add_argument(
        '--local',
        type=ast.literal_eval,
        default=True,
        help='Whether to run as local mode.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help="The device type.")
    parser.add_argument(
        '--sync', type=ast.literal_eval, default=True, help="sync mode.")
    parser.add_argument(
        "--enable_ce",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to run the task "
        "for continuous evaluation.")

    args = parser.parse_args()
    # Append args related to dict
    src_dict = reader.DataReader.load_dict(args.src_vocab_fpath)
    trg_dict = reader.DataReader.load_dict(args.trg_vocab_fpath)
    dict_args = [
        "src_vocab_size", str(len(src_dict)), "trg_vocab_size",
        str(len(trg_dict)), "bos_idx", str(src_dict[args.special_token[0]]),
        "eos_idx", str(src_dict[args.special_token[1]]), "unk_idx",
        str(src_dict[args.special_token[2]])
    ]
    merge_cfg_from_list(args.opts + dict_args,
                        [TrainTaskConfig, ModelHyperParams])
    return args


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
                 util_input_names, sum_cost, token_num):
    # Context to do validation.
    test_program = train_progm.clone()
    with fluid.program_guard(test_program):
        test_program = fluid.io.get_inference_program([avg_cost])

    val_data = reader.DataReader(
        src_vocab_fpath=args.src_vocab_fpath,
        trg_vocab_fpath=args.trg_vocab_fpath,
        fpattern=args.val_file_pattern,
        token_delimiter=args.token_delimiter,
        use_token_batch=args.use_token_batch,
        batch_size=args.batch_size * (1 if args.use_token_batch else dev_count),
        pool_size=args.pool_size,
        sort_type=args.sort_type,
        start_mark=args.special_token[0],
        end_mark=args.special_token[1],
        unk_mark=args.special_token[2],
        # count start and end tokens out
        max_length=ModelHyperParams.max_length - 2,
        clip_last_batch=False,
        shuffle=False,
        shuffle_batch=False)

    test_exe = fluid.ParallelExecutor(
        use_cuda=TrainTaskConfig.use_gpu,
        main_program=test_program,
        share_vars_from=train_exe)

    def test(exe=test_exe):
        test_total_cost = 0
        test_total_token = 0
        test_data = read_multiple(
            reader=val_data.batch_generator,
            count=dev_count if args.use_token_batch else 1)
        for batch_id, data in enumerate(test_data()):
            feed_list = []
            for place_id, data_buffer in enumerate(
                    split_data(
                        data, num_part=dev_count)):
                data_input_dict, _ = prepare_batch_input(
                    data_buffer, data_input_names, util_input_names,
                    ModelHyperParams.eos_idx, ModelHyperParams.eos_idx,
                    ModelHyperParams.n_head, ModelHyperParams.d_model)
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
        fluid.io.load_persistables(exe, TrainTaskConfig.ckpt_path)
        lr_scheduler.current_steps = TrainTaskConfig.start_step
    else:
        print "init fluid.framework.default_startup_program"
        exe.run(fluid.framework.default_startup_program())

    train_data = reader.DataReader(
        src_vocab_fpath=args.src_vocab_fpath,
        trg_vocab_fpath=args.trg_vocab_fpath,
        fpattern=args.train_file_pattern,
        token_delimiter=args.token_delimiter,
        use_token_batch=args.use_token_batch,
        batch_size=args.batch_size * (1 if args.use_token_batch else dev_count),
        pool_size=args.pool_size,
        sort_type=args.sort_type,
        shuffle=args.shuffle,
        shuffle_batch=args.shuffle_batch,
        start_mark=args.special_token[0],
        end_mark=args.special_token[1],
        unk_mark=args.special_token[2],
        # count start and end tokens out
        max_length=ModelHyperParams.max_length - 2,
        clip_last_batch=False)
    train_data = read_multiple(
        reader=train_data.batch_generator,
        count=dev_count if args.use_token_batch else 1)

    build_strategy = fluid.BuildStrategy()
    # Since the token number differs among devices, customize gradient scale to
    # use token average cost among multi-devices. and the gradient scale is
    # `1 / token_number` for average cost.
    build_strategy.gradient_scale_strategy = fluid.BuildStrategy.GradientScaleStrategy.Customized
    train_exe = fluid.ParallelExecutor(
        use_cuda=TrainTaskConfig.use_gpu,
        loss_name=sum_cost.name,
        main_program=train_progm,
        build_strategy=build_strategy)

    data_input_names = encoder_data_input_fields + decoder_data_input_fields[:
                                                                             -1] + label_data_input_fields

    if args.val_file_pattern is not None:
        test = test_context(train_progm, avg_cost, train_exe, dev_count,
                            data_input_names, util_input_names, sum_cost,
                            token_num)

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
            feed_list = []
            total_num_token = 0
            if args.local:
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
                if args.local:
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
            total_sum_cost = sum_cost_val.sum(
            )  # sum the cost from multi-devices
            total_token_num = token_num_val.sum()
            total_avg_cost = total_sum_cost / total_token_num
            print("epoch: %d, batch: %d, avg loss: %f, normalized loss: %f,"
                  " ppl: %f" % (pass_id, batch_id, total_avg_cost,
                                total_avg_cost - loss_normalizer,
                                np.exp([min(total_avg_cost, 100)])))
            if batch_id > 0 and batch_id % 1000 == 0:
                fluid.io.save_persistables(
                    exe,
                    os.path.join(TrainTaskConfig.ckpt_dir, "latest.checkpoint"))
            init = True

        time_consumed = time.time() - pass_start_time
        # Validate and save the model for inference.
        if args.val_file_pattern is not None:
            val_avg_cost, val_ppl = test()
            print(
                "epoch: %d, val avg loss: %f, val normalized loss: %f, val ppl: %f,"
                " consumed %fs" % (pass_id, val_avg_cost,
                                   val_avg_cost - loss_normalizer, val_ppl,
                                   time_consumed))
        else:
            print("epoch: %d, consumed %fs" % (pass_id, time_consumed))
        fluid.io.save_persistables(
            exe,
            os.path.join(TrainTaskConfig.ckpt_dir,
                         "pass_" + str(pass_id) + ".checkpoint"))
        fluid.io.save_inference_model(
            os.path.join(TrainTaskConfig.model_dir,
                         "pass_" + str(pass_id) + ".infer.model"),
            data_input_names[:-2], [predict], exe)
    if args.enable_ce:  # For CE
        print("kpis\ttrain_cost_card%d\t%f" % (dev_count, total_avg_cost))
        print("kpis\ttest_cost_card%d\t%f" % (dev_count, val_avg_cost))
        print("kpis\ttrain_duration_card%d\t%f" % (dev_count, time_consumed))


def train(args):
    # priority: ENV > args > config
    is_local = os.getenv("PADDLE_IS_LOCAL", "1")
    if is_local == '0':
        args.local = False
    print args

    if args.device == 'CPU':
        TrainTaskConfig.use_gpu = False

    training_role = os.getenv("TRAINING_ROLE", "TRAINER")

    if training_role == "PSERVER" or (not TrainTaskConfig.use_gpu):
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    else:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()

    exe = fluid.Executor(place)

    if args.enable_ce:
        fluid.default_startup_program().random_seed = 1000

    sum_cost, avg_cost, predict, token_num = transformer(
        ModelHyperParams.src_vocab_size, ModelHyperParams.trg_vocab_size,
        ModelHyperParams.max_length + 1, ModelHyperParams.n_layer,
        ModelHyperParams.n_head, ModelHyperParams.d_key,
        ModelHyperParams.d_value, ModelHyperParams.d_model,
        ModelHyperParams.d_inner_hid, ModelHyperParams.dropout,
        ModelHyperParams.weight_sharing, TrainTaskConfig.label_smooth_eps)
    lr_scheduler = LearningRateScheduler(ModelHyperParams.d_model,
                                         TrainTaskConfig.warmup_steps,
                                         TrainTaskConfig.learning_rate)

    if args.local:
        optimizer = fluid.optimizer.Adam(
            learning_rate=lr_scheduler.learning_rate,
            beta1=TrainTaskConfig.beta1,
            beta2=TrainTaskConfig.beta2,
            epsilon=TrainTaskConfig.eps)
        optimizer.minimize(sum_cost)
    elif args.sync == False:
        optimizer = fluid.optimizer.SGD(0.003)
        optimizer.minimize(sum_cost)
    else:
        lr_decay = fluid.layers\
         .learning_rate_scheduler\
         .noam_decay(ModelHyperParams.d_model,
            TrainTaskConfig.warmup_steps)

        optimizer = fluid.optimizer.Adam(
            learning_rate=lr_decay,
            beta1=TrainTaskConfig.beta1,
            beta2=TrainTaskConfig.beta2,
            epsilon=TrainTaskConfig.eps)
        optimizer.minimize(sum_cost)

    if args.local:
        print("local start_up:")
        train_loop(exe,
                   fluid.default_main_program(), dev_count, sum_cost, avg_cost,
                   lr_scheduler, token_num, predict)
    else:
        port = os.getenv("PADDLE_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVERS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        trainers = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))
        current_endpoint = os.getenv("POD_IP") + ":" + port
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        t = fluid.DistributeTranspiler()
        t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)

        if training_role == "PSERVER":
            current_endpoint = os.getenv("POD_IP") + ":" + os.getenv(
                "PADDLE_PORT")
            if not current_endpoint:
                print("need env SERVER_ENDPOINT")
                exit(1)
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)

            print "psserver begin run"
            with open('pserver_startup.desc', 'w') as f:
                f.write(str(pserver_startup))
            with open('pserver_prog.desc', 'w') as f:
                f.write(str(pserver_prog))
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":

            trainer_prog = t.get_trainer_program()
            with open('trainer_prog.desc', 'w') as f:
                f.write(str(trainer_prog))
            train_loop(exe, trainer_prog, dev_count, sum_cost, avg_cost,
                       lr_scheduler, token_num, predict)
        else:
            print("environment var TRAINER_ROLE should be TRAINER os PSERVER")


if __name__ == "__main__":
    args = parse_args()
    train(args)
