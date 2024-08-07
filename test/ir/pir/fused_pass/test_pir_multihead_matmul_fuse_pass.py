# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from pass_test import PassTest

import paddle
from paddle.base import core
from paddle.pir.core import create_parameter

np.random.seed(42)
paddle.enable_static()


class TestVitAttentionPattern(PassTest):
    r'''
    x        w
    |        |
      matmul       bias
        |           |
        elementwise_add
             |
            reshape
              |
           transpose
      /       |       \
   slice    slice     slice
     |        |         |
     |        |      transpose
     |        |         |
     |           matmul
     |             |
     |           scale
     |             |
     |           softmax
     \             /
       \         /
          matmul
            |
         transpose
            |
          reshape
    '''

    def is_program_valid(self, program):
        return True

    def build_ir_program(self):
        for bs in [1]:
            for seq_len in [128]:
                for head_dim in [64]:
                    for num_heads in [12]:
                        with paddle.pir_utils.IrGuard():
                            main_prog = paddle.static.Program()
                            start_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(
                                main_prog, start_prog
                            ):
                                hidden_dim = head_dim * num_heads
                                x = paddle.static.data(
                                    name='x',
                                    shape=[bs, seq_len, hidden_dim],
                                    dtype='float32',
                                )
                                bias = paddle.static.data(
                                    name='bias',
                                    shape=[3 * hidden_dim],
                                    dtype='float32',
                                )

                                w = create_parameter(
                                    name="w",
                                    shape=[hidden_dim, 3 * hidden_dim],
                                    dtype='float32',
                                    initializer=paddle.nn.initializer.Assign(
                                        np.random.rand(
                                            hidden_dim, 3 * hidden_dim
                                        ).astype(np.float32)
                                    ),
                                )
                                matmul_out_1 = paddle.matmul(x, w)
                                add_out = paddle.add(matmul_out_1, bias)
                                # bs,seq_len,num_heads,3,head_dim
                                reshape_out_1 = paddle.reshape(
                                    add_out,
                                    shape=[bs, seq_len, 3, num_heads, head_dim],
                                )
                                transpose_out_1 = paddle.transpose(
                                    reshape_out_1, perm=[2, 0, 3, 1, 4]
                                )
                                # bs,num_heads,seq_len,head_dim
                                q = transpose_out_1[0, :, :, :, :]
                                k = transpose_out_1[1, :, :, :, :]
                                v = transpose_out_1[2, :, :, :, :]
                                matmul_out_2 = paddle.matmul(
                                    q, paddle.transpose(k, perm=[0, 1, 3, 2])
                                )
                                scale_out = paddle.scale(
                                    matmul_out_2,
                                    scale=0.125,
                                    bias=0.0,
                                )
                                softmax_out = paddle.nn.functional.softmax(
                                    scale_out
                                )
                                # bs,num_head,seq_len,head_dim
                                matmul_out_3 = paddle.matmul(softmax_out, v)
                                transpose_out_2 = paddle.transpose(
                                    matmul_out_3, perm=[0, 2, 1, 3]
                                )
                                reshape_out_2 = paddle.reshape(
                                    transpose_out_2,
                                    shape=[bs, seq_len, num_heads * head_dim],
                                )
                                out = paddle.assign(reshape_out_2)
                                self.pass_attr_list = [
                                    {'multihead_matmul_fuse_pass': {}}
                                ]
                                self.feeds = {
                                    "x": np.random.random(
                                        (bs, seq_len, hidden_dim)
                                    ).astype("float32")
                                    - 0.5,
                                    "bias": np.random.random(
                                        3 * hidden_dim
                                    ).astype("float32"),
                                }
                                self.fetch_list = [out]
                                self.valid_op_map = {
                                    "pd_op.multihead_matmul": 1,
                                }
                                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
