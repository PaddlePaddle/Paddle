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
from paddle import base
from paddle.base import core

paddle.enable_static()


class TestCollectiveAllreduce(TestCollectiveRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program):
        ring_id = 0
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(
                name="tindata", shape=[-1, 10, 1000], dtype='float32'
            )
            tindata.desc.set_need_check_feed(False)
            toutdata = main_prog.current_block().create_var(
                name="outofallreduce",
                dtype='float32',
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=False,
            )

            # tout = tin + tin - tin = tin
            if True:
                main_prog.global_block().append_op(
                    type="elementwise_add",
                    inputs={
                        'X': tindata,
                        'Y': tindata,
                    },
                    outputs={'Out': toutdata},
                )
                main_prog.global_block().append_op(
                    type="elementwise_sub",
                    inputs={
                        'X': toutdata,
                        'Y': tindata,
                    },
                    outputs={'Out': toutdata},
                )

            main_prog.global_block().append_op(
                type='c_wait_compute',
                inputs={'X': toutdata},
                outputs={'Out': toutdata},
                attrs={'ring_id': ring_id},
            )

            main_prog.global_block().append_op(
                type="c_allreduce_sum",
                inputs={'X': toutdata},
                attrs={'ring_id': ring_id},
                outputs={'Out': toutdata},
                attr={'use_calc_stream': False},
            )

            main_prog.global_block().append_op(
                type="c_wait_comm",
                inputs={'X': toutdata},
                outputs={'Out': toutdata},
                attrs={'ring_id': ring_id},
            )

            # tout = tin + tout - tin = tout
            if True:
                main_prog.global_block().append_op(
                    type="elementwise_add",
                    inputs={
                        'X': tindata,
                        'Y': toutdata,
                    },
                    outputs={'Out': toutdata},
                )
                main_prog.global_block().append_op(
                    type="elementwise_sub",
                    inputs={
                        'X': toutdata,
                        'Y': tindata,
                    },
                    outputs={'Out': toutdata},
                )

            return toutdata

    def get_model_new_comm(self, main_prog, startup_program, dtype="float32"):
        return self.get_model(main_prog, startup_program)


if __name__ == "__main__":
    runtime_main(TestCollectiveAllreduce, "allreduce", 0)
