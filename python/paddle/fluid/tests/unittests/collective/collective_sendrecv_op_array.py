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

import numpy as np
from test_collective_base import TestCollectiveRunnerBase, runtime_main

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers

paddle.enable_static()


class TestCollectiveSendRecv(TestCollectiveRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program):
        ring_id = self.global_ring_id
        with fluid.program_guard(main_prog, startup_program):
            tindata = layers.data(
                name="tindata",
                shape=[10, 1000],
                dtype='float64',
                append_batch_size=False,
            )
            if self.rank == 0:
                data1 = fluid.layers.assign(
                    np.array([[0, 1, 2]], dtype='float32')
                )
                data2 = fluid.layers.assign(
                    np.array([[3, 4, 5]], dtype='float32')
                )
            elif self.rank == 1:
                data1 = fluid.layers.assign(
                    np.array([[3, 4, 5]], dtype='float32')
                )
                data2 = fluid.layers.assign(
                    np.array([[0, 1, 2]], dtype='float32')
                )
            tensor_array = paddle.tensor.create_array(dtype='float32')
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
            paddle.tensor.array_write(data1, i, tensor_array)
            paddle.tensor.array_write(data2, i + 1, tensor_array)
            if self.rank == 0:
                main_prog.global_block().append_op(
                    type="send_v2",
                    inputs={'X': tensor_array},
                    attrs={
                        'ring_id': ring_id,
                        'peer': 1,
                        'use_calc_stream': True,
                    },
                )
            else:
                main_prog.global_block().append_op(
                    type="recv_v2",
                    outputs={'Out': tensor_array},
                    attrs={
                        'peer': 0,
                        'ring_id': ring_id,
                        'dtype': data1.dtype,
                        'out_shape': [1, 3],
                        'use_calc_stream': True,
                    },
                )
            return tensor_array


if __name__ == "__main__":
    runtime_main(TestCollectiveSendRecv, "sendrecv_array", 0)
