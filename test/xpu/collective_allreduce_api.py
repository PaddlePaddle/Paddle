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


def all_reduce_new(tensor, reduce_type=str(dist.ReduceOp.SUM), group=None):
    op_type = 'all_reduce'
    data_feeder.check_variable_and_dtype(
        tensor,
        'tensor',
        [
            'float16',
            'float32',
            'int32',
        ],
        op_type,
    )

    ring_id = 0 if group is None else group.id

    if not isinstance(ring_id, int):
        raise ValueError("The type of 'ring_id' for all_reduce should be int.")

    # TODO: Support task and use task.wait in static graph mode
    #       Use use_calc_stream rather than sync_op
    helper = framework.LayerHelper(op_type, **locals())
    if not reduce_type.isdigit():
        raise ValueError(
            "The type of 'reduce_type' for all_reduce should be int."
        )
    helper.append_op(
        type=op_type,
        inputs={'x': [tensor]},
        outputs={'out': [tensor]},
        attrs={'ring_id': ring_id, 'reduce_type': int(reduce_type)},
    )


class TestCollectiveAllreduceAPI(test_base.TestCollectiveAPIRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, dtype='float32'):
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(
                name="tindata", shape=[10, 1000], dtype=dtype
            )
            reduce_type = int(os.getenv("REDUCE_TYPE"))
            paddle.distributed.all_reduce(tindata, op=reduce_type)
            return [tindata]

    def get_model_new(
        self,
        main_prog,
        startup_program,
        rank,
        dtype='float32',
        reduce_type=str(dist.ReduceOp.SUM),
    ):
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(
                name="tindata", shape=[10, 1000], dtype=dtype
            )
            all_reduce_new(tindata, reduce_type)
            return [tindata]


if __name__ == "__main__":
    test_base.runtime_main(TestCollectiveAllreduceAPI, "allreduce")
