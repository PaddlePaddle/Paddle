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

paddle.enable_static()


class TestVitAttentionPattern(PassTest):
    r'''
            input
              |
           transpose
        /     |     \
     slice  slice  slice
       |      |      |
       |  transpose  |
        \     |      |
         matmul      |
           |         |
        softmax      |
           |____ ____/
                |
             matmul
                |
            transpose
                |
               out
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
                                input = paddle.static.data(
                                    name='input',
                                    shape=[bs, seq_len, 3, num_heads, head_dim],
                                    dtype='float32',
                                )

                                # bs,seq_len,num_heads,3,head_dim
                                transpose_out_1 = paddle.transpose(
                                    input, perm=[2, 0, 3, 1, 4]
                                )
                                # bs,num_heads,seq_len,head_dim
                                q = transpose_out_1[0, :, :, :, :]
                                k = transpose_out_1[1, :, :, :, :]
                                v = transpose_out_1[2, :, :, :, :]
                                matmul_out_2 = paddle.matmul(
                                    q, paddle.transpose(k, perm=[0, 1, 3, 2])
                                )

                                softmax_out = paddle.nn.functional.softmax(
                                    matmul_out_2
                                )
                                # bs,num_head,seq_len,head_dim
                                matmul_out_3 = paddle.matmul(softmax_out, v)
                                transpose_out_2 = paddle.transpose(
                                    matmul_out_3, perm=[0, 2, 1, 3]
                                )
                                out = paddle.assign(transpose_out_2)
                                self.pass_attr_list = [
                                    {'self_attention_fuse_pass': {}}
                                ]
                                self.feeds = {
                                    "input": np.random.random(
                                        (bs, seq_len, 3, num_heads, head_dim)
                                    ).astype("float32")
                                    - 0.5
                                }
                                self.fetch_list = [out]
                                self.valid_op_map = {
                                    "pd_op.self_dp_attention": 1,
                                    "pd_op.matmul": 0,
                                    "pd_op.transpose": 0,
                                    "pd_op.softmax": 0,
                                    "pd_op.slice": 0,
                                }
                                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
