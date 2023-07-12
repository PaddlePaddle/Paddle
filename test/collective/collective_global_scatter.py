# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from legacy_test.test_collective_api_base import (
    TestCollectiveAPIRunnerBase,
    dump_output,
    runtime_main,
)

import paddle
from paddle import fluid
from paddle import framework
import paddle.distributed as dist
from paddle.fluid import data_feeder
from paddle.distributed.utils import moe_utils

paddle.enable_static()

def alltoall(
    in_tensor_or_tensor_list,
    out_tensor_or_tensor_list,
    group=None,
    sync_op=True,
    use_calc_stream=False,
):
    op_type = 'alltoall'
    ring_id = 0 if group is None else group.id
    nranks = dist.get_world_size()
    helper = framework.LayerHelper(op_type, **locals())

    in_tensor = in_tensor_or_tensor_list
    if isinstance(in_tensor_or_tensor_list, list):
        if len(in_tensor_or_tensor_list) == 0:
            raise RuntimeError("The input tensor_list should not be empty.")
        # 0-D use stack/unstack while others use concat/split
        if len(in_tensor_or_tensor_list[0].shape) == 0:
            in_tensor = paddle.stack(in_tensor_or_tensor_list, axis=0)
        else:
            in_tensor = paddle.concat(in_tensor_or_tensor_list, axis=0)

    out_tensor = out_tensor_or_tensor_list
    if isinstance(out_tensor_or_tensor_list, list):
        if len(out_tensor_or_tensor_list) != 0:
            raise ValueError(
                "The 'out_tensor_list' for all_to_all " "must be an empty list."
            )
        out_tensor = helper.create_variable_for_type_inference(
            dtype=in_tensor.dtype
        )

    data_feeder.check_variable_and_dtype(
        in_tensor,
        'in_tensor',
        ['float16', 'float32', 'float64', 'int32', 'int64'],
        'all_to_all',
    )
    helper.append_op(
        type=op_type,
        inputs={'X': [in_tensor]},
        outputs={'Out': [out_tensor]},
        attrs={
            'ring_id': ring_id,
            'use_calc_stream': sync_op 
        },
    )
    # NOTE(liyurui): If the argument `out_tensor_or_tensor_list` is a tensor_list,
    # we need to split the result. So we should wait the result of all_to_all
    # before split if the communication is not on calc stream.
    if isinstance(out_tensor_or_tensor_list, list):
        if not sync_op:
            dist.wait(out_tensor, use_calc_stream=False)
        # 0-D use stack/unstack while others use concat/split
        if len(in_tensor_or_tensor_list[0].shape) == 0:
            out_tensor_or_tensor_list.extend(paddle.unstack(out_tensor, 0))
        else:
            out_tensor_or_tensor_list.extend(
                paddle.split(out_tensor, nranks, 0)
            )
    return None

class TestCollectiveGlobalScatterAPI(TestCollectiveAPIRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        with fluid.program_guard(main_prog, startup_program):
            seed = os.getpid()
            np.random.seed(seed)
            in_feat = 2
            n_expert = 2
            world_size = 2
            tot_expert = n_expert * world_size
            local_input_buf = paddle.static.data(
                name="local_input_buf", shape=[-1, in_feat], dtype="float32"
            )
            local_expert_count = paddle.static.data(
                name="local_expert_count", shape=[tot_expert], dtype="int64"
            )
            global_expert_count = []
            alltoall(
                paddle.split(local_expert_count, 2, axis=0), global_expert_count
            )
            global_expert_count = paddle.concat(global_expert_count, axis=0)
            output = moe_utils.global_scatter(
                local_input_buf, local_expert_count, global_expert_count
            )
            return [output]

    def run_trainer(self, args):
        train_prog = fluid.Program()
        startup_prog = fluid.Program()
        endpoints = args["endpoints"].split(",")
        rank = args["trainerid"]
        current_endpoint = args["currentendpoint"]
        nranks = 2
        paddle.distributed.init_parallel_env()
        if args['backend'] == 'nccl':
            device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
            place = fluid.CUDAPlace(
                device_id
            )  # if args.use_gpu else fluid.CPUPlace()
        elif args['backend'] == 'bkcl':
            device_id = int(os.getenv("FLAGS_selected_xpus", "0"))
            place = fluid.XPUPlace(device_id)
        else:
            place = fluid.CPUPlace()
        np.random.seed(os.getpid())
        in_feat = 2
        n_expert = 2
        world_size = 2
        tot_expert = n_expert * world_size
        local_expert_count = np.random.randint(1, 4, size=tot_expert).astype(
            "int64"
        )
        fwd_expert_count = sum(local_expert_count)
        local_input_buf = np.random.rand(fwd_expert_count, in_feat).astype(
            "float32"
        )
        if args['static_mode']:
            result = self.get_model(train_prog, startup_prog, rank)
            exe = fluid.Executor(place)
            exe.run(startup_prog)
            fetch_list = []
            for elem in result:
                fetch_list.append(elem.name)
            out = exe.run(
                train_prog,
                feed={
                    'local_expert_count': local_expert_count,
                    'local_input_buf': local_input_buf,
                },
                fetch_list=fetch_list,
            )

        dump_output(out)


if __name__ == "__main__":
    runtime_main(TestCollectiveGlobalScatterAPI, "global_scatter")
