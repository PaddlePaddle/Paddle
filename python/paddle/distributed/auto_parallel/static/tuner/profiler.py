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

import argparse
import json
import os
import pickle
import sys
import time
import traceback

import paddle
from paddle.distributed.auto_parallel.static.dist_loader import (
    DistributedDataLoaderFromGenerator,
)
from paddle.distributed.auto_parallel.static.process_group import (
    get_all_process_groups,
    new_process_group,
)
from paddle.distributed.collective import _get_global_env
from paddle.framework import Program, _current_expected_place
from paddle.static import Operator

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
        help="integer indicates the warmup step before starting profile.",
    )
    parser.add_argument(
        "--profile_end_step",
        default=30,
        type=int,
        help="integer indicates at the end step of profile.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        help="the rank id of the this process.",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        required=True,
        help="the device id of the this process.",
    )
    parser.add_argument(
        "--ctx_filename",
        type=str,
        required=True,
        help="the filename to the profile context file saved by optimization tuner",
    )

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
        print(process_group)
        process_group.instantiate()


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


def create_dataloader(
    main_program, startup_program, profile_ctx, epochs=1, steps_per_epoch=None
):

    dataset = profile_ctx["dataset"]
    main_block = main_program.global_block()
    feed_list = []
    for name in dataset.input_names:
        if name in main_block.vars:
            feed_list.append(main_block.vars[name])

    # remove the first three ops if multi run fit/evaluate/predict
    op_size = len(main_block.ops)
    if main_block.ops[0].type == 'create_py_reader':
        op_size -= 3
        for _ in range(3):
            main_block._remove_op(0, sync=False)

    # insert read op at the end of program
    places = paddle.static.cuda_places()
    with paddle.static.program_guard(main_program, startup_program):
        dataloader = DistributedDataLoaderFromGenerator(
            dataset=dataset,
            feed_list=feed_list,
            capacity=70,
            places=places,
            batch_size=dataset.batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            data_parallel_world_size=dataset.dp_world_size,
            data_parallel_rank=dataset.dp_rank,
        )

    # move read op from the end of program to the start of program
    new_op_size = len(main_block.ops)
    for _ in range(new_op_size - 1, op_size - 1, -1):
        op = main_block.ops[new_op_size - 1]
        new_op_desc = main_block.desc._prepend_op()
        new_op_desc.copy_from(op.desc)
        new_op = Operator(main_block, new_op_desc, type=new_op_desc.type())
        main_block.ops.insert(0, new_op)
    for _ in range(new_op_size - op_size):
        main_block._remove_op(new_op_size, sync=False)
    main_block._sync_with_cpp()
    return dataloader


def init_comm(profile_ctx):
    # override the env for current process
    dist_env = profile_ctx['distributed_env']
    genv = _get_global_env()
    genv = dist_env
    print(
        "current process rank: {}, device_id: {}, ip: {}.".format(
            genv.rank,
            genv.device_id,
            genv.current_endpoint,
        )
    )

    # init nccl comm
    group_map = profile_ctx['group_map']
    init_process_groups(group_map, args.rank)


def load_programs(profile_ctx):
    main_program_desc_str = profile_ctx['main_program_decs']
    main_program = Program.parse_from_string(main_program_desc_str)

    startup_program_decs_str = profile_ctx['startup_program_decs']
    startup_program = Program.parse_from_string(startup_program_decs_str)

    loss_var_name = profile_ctx["loss_var_name"]
    assert main_program.global_block().has_var(loss_var_name)
    loss_var = main_program.global_block().var(loss_var_name)

    return main_program, startup_program, loss_var


def get_executor():
    place_type = _current_expected_place()
    if not isinstance(place_type, paddle.CUDAPlace):
        raise RuntimeError("OptimizationTuner only support CUDA GPU right now.")

    genv = _get_global_env()
    place = paddle.CUDAPlace(genv.device_id)
    exe = paddle.static.Executor(place)
    return exe


def profiler(args):
    """
    main function to profile experiment for each pass hyper-parameter.
    """
    # load ctx
    if not os.path.isfile(args.ctx_filename):
        raise ValueError(
            f"There is no profile context named {args.ctx_filename}."
        )
    with open(args.ctx_filename, 'rb') as f:
        profile_ctx = pickle.load(f, encoding='latin1')

    init_comm(profile_ctx)

    main_program, startup_program, loss_var = load_programs(profile_ctx)

    data_loader = create_dataloader(main_program, startup_program, profile_ctx)

    result_path = profile_ctx["result_filename"]

    exe = get_executor()

    try:
        exe.run(startup_program)
        # profile main
        duration = 0
        eval_step = 0
        data_loader._inner_dataloader.start()
        while eval_step < args.profile_end_step:
            start_time = time.time()

            loss = exe.run(
                main_program,
                fetch_list=[loss_var],
                use_program_cache=True,
            )

            end_time = time.time()

            if eval_step >= args.profile_start_step:
                duration += end_time - start_time

            print("step: %d, loss_print: %f" % (eval_step, loss[0]))
            eval_step += 1

        avg_tput = (
            1.0 * (args.profile_end_step - args.profile_start_step) / duration
        )

        result_dict = {
            "Throughtput": avg_tput,
            "ErrorType": None,
        }

        if paddle.distributed.get_rank() == 0:
            with open(result_path, 'w') as fp:
                json.dump(result_dict, fp)

        print(f"profile done! avg speed : {avg_tput} step / s.")

    except paddle.framework.core.EOFException:
        data_loader._inner_dataloader.reset()

    except Exception as e:

        error_type = get_cpp_error_type(e)
        result_dict = {
            "Throughtput": -1,
            "ErrorType": error_type,
        }
        if not os.path.isfile(result_path):
            with open(result_path, 'w') as fp:
                json.dump(result_dict, fp)

        print(f"profile failed with error: [{error_type}]")
        print(e)
        print(traceback.format_exc())

        data_loader._inner_dataloader.reset()
        del data_loader._inner_dataloader
        sys.exit(1)

    data_loader._inner_dataloader.reset()
    del data_loader._inner_dataloader


if __name__ == "__main__":
    args = parse_args()
    profiler(args)
