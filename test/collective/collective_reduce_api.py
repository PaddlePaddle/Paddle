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

import sys

sys.path.append("../legacy_test")

from test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main

import paddle
import paddle.distributed as dist
from paddle import fluid, framework
from paddle.distributed.communication.reduce import _get_reduce_op
from paddle.fluid import data_feeder

paddle.enable_static()


def reduce(tensor, dst, op=dist.ReduceOp.SUM, group=None, sync_op=True):
    op_type = _get_reduce_op(op, "reduce")
    data_feeder.check_variable_and_dtype(
        tensor,
        'tensor',
        [
            'float16',
            'float32',
            'float64',
            'int32',
            'int64',
            'int8',
            'uint8',
            'bool',
            'uint16',
        ],
        op_type,
    )

    ring_id = 0 if group is None else group.id

    helper = framework.LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
        attrs={
            'ring_id': ring_id,
            'root_id': dst,
            'use_calc_stream': sync_op,
        },
    )
    return None


class TestCollectiveReduceAPI(TestCollectiveAPIRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank):
        with fluid.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(
                name="tindata", shape=[-1, 10, 1000], dtype='float32'
            )
            tindata.desc.set_need_check_feed(False)
            reduce(tindata, dst=0)
            return [tindata]

    def get_model_new(
        self,
        main_prog,
        startup_program,
        rank,
        dtype='float32',
        reduce_type=dist.ReduceOp.SUM,
    ):
        with fluid.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(
                name="tindata", shape=[10, 1000], dtype=dtype
            )
            tindata.desc.set_need_check_feed(False)
            paddle.distributed.reduce(tindata, dst=0, op=int(reduce_type))
            return [tindata]


if __name__ == "__main__":
    runtime_main(TestCollectiveReduceAPI, "reduce")
