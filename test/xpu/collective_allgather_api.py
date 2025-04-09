# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os

import test_collective_api_base as test_base

import paddle
import paddle.distributed as dist
from paddle import base, framework
from paddle.base import data_feeder

paddle.enable_static()


def all_gather_new(tensor_list, tensor, group=None):
    op_type = 'all_gather'
    helper = framework.LayerHelper(op_type, **locals())
    out = helper.create_variable_for_type_inference(dtype=tensor.dtype)
    for elem in tensor_list:
        data_feeder.check_variable_and_dtype(
            elem,
            'tensor_list',
            [
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
            ],
            op_type,
        )
    data_feeder.check_variable_and_dtype(
        tensor,
        'tensor',
        [
            'float16',
            'float32',
            'float64',
            'int32',
            'int64',
        ],
        op_type,
    )

    ring_id = 0 if group is None else group.id
    nranks = dist.get_world_size()
    helper.append_op(
        type=op_type,
        inputs={'x': [tensor]},
        outputs={'out': [out]},
        attrs={
            'ring_id': ring_id,
            'nranks': nranks,
        },
    )
    tensor_list.clear()
    tensor_list.extend(paddle.split(out, nranks, 0))


class TestCollectiveAllgatherAPI(test_base.TestCollectiveAPIRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, dtype="float32"):
        with base.program_guard(main_prog, startup_program):
            tensor_list = []
            tindata = paddle.static.data(
                name="tindata", shape=[10, 1000], dtype=dtype
            )
            paddle.distributed.all_gather(tensor_list, tindata)
            return tensor_list

    def get_model_new(
        self, main_prog, startup_program, rank, dtype=None, reduce_type=None
    ):
        with base.program_guard(main_prog, startup_program):
            tensor_list = []
            tindata = paddle.static.data(
                name="tindata", shape=[10, 1000], dtype=dtype
            )
            all_gather_new(tensor_list, tindata)
            return tensor_list

    def run_trainer(self, args):
        train_prog = base.Program()
        startup_prog = base.Program()
        endpoints = args["endpoints"].split(",")
        rank = args["trainerid"]
        current_endpoint = args["currentendpoint"]
        nranks = 2

        paddle.distributed.collective._init_parallel_env(args["backend"])

        if args['backend'] == 'nccl':
            device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
            place = base.CUDAPlace(
                device_id
            )  # if args.use_gpu else base.CPUPlace()
        elif args['backend'] == 'bkcl':
            device_id = int(os.getenv("FLAGS_selected_xpus", "0"))
            place = base.XPUPlace(device_id)
        else:
            place = base.CPUPlace()
        indata = test_base.create_test_data(
            shape=(10, 1000), dtype=args["dtype"], seed=os.getpid()
        )
        assert (
            args['static_mode'] == 1
        ), "collective_allgather_api only support static graph mode"
        result = (
            self.get_model_new(
                train_prog, startup_prog, rank, dtype=args["dtype"]
            )
            if args["use_comm_context"]
            else self.get_model(
                train_prog, startup_prog, rank, dtype=args["dtype"]
            )
        )
        exe = base.Executor(place)
        exe.run(startup_prog)
        fetch_list = []
        for elem in result:
            fetch_list.append(elem.name)
        out = exe.run(
            train_prog, feed={'tindata': indata}, fetch_list=fetch_list
        )
        test_base.dump_output(out)


if __name__ == "__main__":
    test_base.runtime_main(TestCollectiveAllgatherAPI, "allgather")
