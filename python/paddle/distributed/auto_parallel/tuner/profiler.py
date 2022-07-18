# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
parse cmd args & setting

load ctx restore env 


run 

write result
"""
import os
import sys
import argparse
import traceback
import pickle
import json
import time
import numpy as np

import paddle
from paddle.fluid.framework import Program, _current_expected_place
from paddle.distributed.auto_parallel.process_group import clear_all_process_groups, get_all_process_groups, new_process_group
from paddle.distributed.collective import _get_global_env

paddle.enable_static()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile_start_step",
        default=10,
        type=int,
        help="integer indicates the warmup step before starting profile.")
    parser.add_argument("--profile_end_step",
                        default=30,
                        type=int,
                        help="integer indicates at the end step of profile.")
    parser.add_argument("--rank",
                        type=int,
                        required=True,
                        help="the rank id of the this process.")
    parser.add_argument("--device_id",
                        type=int,
                        required=True,
                        help="the device id of the this process.")
    parser.add_argument(
        "--ctx_filename",
        type=str,
        required=True,
        help=
        "the filename to the profile context file saved by optimizaiton tuner")

    args = parser.parse_args()

    return args


def init_process_groups(group_map, rank):
    for group_id, ranks in group_map.items():
        if group_id == 0:
            continue
        new_process_group(ranks=ranks, group_id=group_id)

    # TODO should instantiate global group first
    all_process_groups = get_all_process_groups()
    for process_group in all_process_groups:
        if process_group.id == 0 or rank not in process_group.ranks:
            continue
        print(process_group)
        process_group.instantiate()


def gen_data(batch_size):
    sequence_len = 512
    tokens = []
    position_ids = []
    attention_mask = []
    labels = []
    loss_mask = []
    for _ in range(batch_size):
        tokens.append(np.random.randint(50304, size=sequence_len))
        position_ids.append(np.arange(sequence_len))
        attention_mask.append([np.tril(np.ones(sequence_len))])
        labels.append(np.random.randint(50304, size=sequence_len))
        loss_mask.append(np.ones(sequence_len))

    return tokens, position_ids, attention_mask, labels, loss_mask


def get_cpp_error_type(error):

    msg = str(error).splitlines()
    cpp_error_types = [
        'InvalidArgumentError',
        'NotFoundError',
        'OutOfRangeError',
        'AlreadyExistsError',
        'ResourceExhaustedError',
        'PreconditionNotMetError',
        'PermissionDeniedError',
        'ExecutionTimeoutError',
        'UnimplementedError',
        'UnavailableError',
        'FatalError',
        'ExternalError',
    ]
    error_type = 'FatalError'
    for et in cpp_error_types:
        for line in msg:
            if et in line:
                return et
    return error_type


def profiler():
    """
    main function to profile experiment for each pass hyper-parameter.
    """

    args = parse_args()

    # load ctx
    if not os.path.isfile(args.ctx_filename):
        raise ValueError("There is no profile context named {}.".format(
            args.ctx_filename))
    # if not os.path.isfile(args.main_program_filename):
    #     raise ValueError("There is no main program named {}.".format(args.ctx_filename))
    # if not os.path.isfile(args.startup_program_filename):
    #     raise ValueError("There is no startup program named {}.".format(args.ctx_filename))

    with open(args.ctx_filename, 'rb') as f:
        profile_ctx = pickle.load(f, encoding='latin1')

    # TODO check type and exist
    dist_env = profile_ctx['distributed_env']
    print("sent env: ", dist_env.rank, dist_env.device_id,
          dist_env.trainer_endpoints)
    batch_size = profile_ctx["batch_size"]
    genv = _get_global_env()
    genv = dist_env
    env = _get_global_env()
    print("cur proc env: ", env.device_id, env.rank, env.current_endpoint)
    group_map = profile_ctx['group_map']
    init_process_groups(group_map, args.rank)

    main_program_desc_str = profile_ctx['main_program_decs']
    main_program = Program.parse_from_string(main_program_desc_str)
    # print("=========main_program" * 8)
    # print(main_program)
    # print("=========main_program" * 8)
    # with open(args.main_program_filename, "rb") as f:
    #     main_program_desc_str = f.read()
    #     main_program = Program.parse_from_string(main_program_desc_str)

    loss_var_name = profile_ctx["loss_var_name"]
    assert main_program.global_block().has_var(loss_var_name)
    loss_var = main_program.global_block().var(loss_var_name)

    startup_program_decs_str = profile_ctx['startup_program_decs']
    startup_program = Program.parse_from_string(startup_program_decs_str)

    result_path = profile_ctx["result_filename"]
    # print("=========startup_program" * 8)
    # print(startup_program)
    # print("=========startup_program" * 8)
    # with open(args.startup_program_filename, "rb") as f:
    #     startup_program_desc_str = f.read()
    #     startup_program = Program.parse_from_string(startup_program_desc_str)

    # reconstruct communicator

    # TODO build fake dataloader

    place_type = _current_expected_place()
    if not isinstance(place_type, paddle.CUDAPlace):
        raise NotImplementedError

    place = paddle.CUDAPlace(dist_env.device_id)
    exe = paddle.static.Executor(place)

    exe.run(startup_program)

    # profile main
    duration = 0
    eval_step = 0
    print("batch size: ", batch_size)
    tokens, position_ids, attention_mask, labels, loss_mask = gen_data(
        batch_size)

    try:
        while eval_step < args.profile_end_step:
            start_time = time.time()

            loss = exe.run(
                main_program,
                feed={
                    "tokens": tokens,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "loss_mask": loss_mask
                },
                fetch_list=[loss_var],
                use_program_cache=True,
            )

            end_time = time.time()

            if eval_step >= args.profile_start_step:
                duration += end_time - start_time

            print(loss[0])
            print("step: %d, loss_print: %f" % (eval_step, loss[0]))
            eval_step += 1

        avg_tput = 1.0 * (args.profile_end_step -
                          args.profile_start_step) / duration

        result_dict = {
            "Throughtput": avg_tput,
            "ErrorType": None,
        }

        if paddle.distributed.get_rank() == 0:
            with open(result_path, 'w') as fp:
                json.dump(result_dict, fp)

        print("profile done! avg speed : {} step / s.".format((avg_tput)))
    except Exception as e:
        error_type = get_cpp_error_type(e)
        result_dict = {
            "Throughtput": -1,
            "ErrorType": error_type,
        }
        if not os.path.isfile(result_path):
            with open(result_path, 'w') as fp:
                json.dump(result_dict, fp)

        print("profile failed with error: [{}]".format(error_type), )
        exit(1)


if __name__ == "__main__":
    profiler()
