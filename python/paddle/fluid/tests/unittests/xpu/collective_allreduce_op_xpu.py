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

from __future__ import print_function

import os
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
import paddle.fluid.layers as layers
from test_collective_base_xpu import TestCollectiveRunnerBase, runtime_main

paddle.enable_static()


class TestCollectiveAllReduce(TestCollectiveRunnerBase):

    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program):
        ring_id = 0
        with fluid.program_guard(main_prog, startup_program):
            tindata = layers.data(name="tindata",
                                  shape=[10, 1000],
                                  dtype='float32')
            toutdata = main_prog.current_block().create_var(
                name="outofreduce",
                dtype='float32',
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=False)
            main_prog.global_block().append_op(type="c_allreduce_sum",
                                               inputs={'X': tindata},
                                               attrs={
                                                   'ring_id': ring_id,
                                               },
                                               outputs={'Out': toutdata})
            main_prog.global_block().append_op(type="c_sync_comm_stream",
                                               inputs={'X': toutdata},
                                               outputs={'Out': toutdata},
                                               attrs={'ring_id': ring_id})
            return toutdata


if __name__ == "__main__":
    os.environ["BKCL_PCIE_RING"] = "1"
    runtime_main(TestCollectiveAllReduce, "allreduce", 0)
