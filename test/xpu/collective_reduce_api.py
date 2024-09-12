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

from test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main

import paddle
import paddle.distributed as dist
from paddle import base, framework
from paddle.base import data_feeder

paddle.enable_static()


def reduce_new(tensor, dst, reduce_type=str(dist.ReduceOp.SUM), group=None):
    op_type = "reduce"
    data_feeder.check_variable_and_dtype(
        tensor,
        'tensor',
        [
            'float32',
        ],
        op_type,
    )

    ring_id = 0 if group is None else group.id

    helper = framework.LayerHelper(op_type, **locals())
    if not reduce_type.isdigit():
        raise ValueError("The type of 'reduce_type' for reduce should be int.")
    helper.append_op(
        type=op_type,
        inputs={'x': [tensor]},
        outputs={'out': [tensor]},
        attrs={
            'ring_id': ring_id,
            'root_id': dst,
            'reduce_type': int(reduce_type),
        },
    )


class TestCollectiveReduceAPI(TestCollectiveAPIRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, dtype='float32'):
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(
                name="tindata", shape=[-1, 10, 1000], dtype=dtype
            )
            if not paddle.framework.use_pir_api():
                tindata.desc.set_need_check_feed(False)
            paddle.distributed.reduce(tindata, dst=0)
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
            if not paddle.framework.use_pir_api():
                tindata.desc.set_need_check_feed(False)
            reduce_new(tindata, dst=0, reduce_type=reduce_type)
            return [tindata]


if __name__ == "__main__":
    runtime_main(TestCollectiveReduceAPI, "reduce")
