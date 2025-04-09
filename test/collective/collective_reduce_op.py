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

from legacy_test.test_collective_base import (
    TestCollectiveRunnerBase,
    runtime_main,
)

import paddle
import paddle.distributed as dist
from paddle import base
from paddle.base import core

paddle.enable_static()


class TestCollectiveReduce(TestCollectiveRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program):
        ring_id = 0
        rootid = 1
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(
                name="tindata", shape=[-1, 10, 1000], dtype='float32'
            )
            tindata.desc.set_need_check_feed(False)
            toutdata = main_prog.current_block().create_var(
                name="outofreduce",
                dtype='float32',
                type=core.VarDesc.VarType.DENSE_TENSOR,
                persistable=False,
                stop_gradient=False,
            )
            main_prog.global_block().append_op(
                type="reduce",
                inputs={'x': tindata},
                attrs={
                    'ring_id': ring_id,
                    'root_id': rootid,
                    'reduce_type': int(dist.ReduceOp.SUM),
                },
                outputs={'out': toutdata},
            )
            main_prog.global_block().append_op(
                type="c_sync_comm_stream",
                inputs={'X': toutdata},
                outputs={'Out': toutdata},
                attrs={'ring_id': ring_id},
            )
            return toutdata


if __name__ == "__main__":
    runtime_main(TestCollectiveReduce, "reduce", 0)
