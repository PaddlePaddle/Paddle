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


class TestMatmulTransposeReshapeFusePattern(PassTest):
    r'''
    x       y
     \     /
      matmul
        |
    transpose
        |
     reshape
        |
       out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[8, 12, 128, 128], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[8, 12, 128, 64], dtype='float32'
                )

                matmul_out = paddle.matmul(x, y)
                transpose_out = paddle.transpose(matmul_out, perm=[0, 2, 1, 3])
                reshape_out = paddle.reshape(
                    transpose_out, shape=[0, 0, 12 * 64]
                )
                reshape_out = paddle.assign(reshape_out)
                self.pass_list = ['matmul_transpose_reshape_fuse_pass']
                self.feeds = {
                    "x": np.random.random((8, 12, 128, 128)).astype("float32"),
                    "y": np.random.random((8, 12, 128, 64)).astype("float32"),
                }
                self.fetch_list = [reshape_out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.matmul": 0,
                    "pd_op.reshape": 0,
                    "pd_op.transpose": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestMatmulTransposeReshapeAddFusePattern(PassTest):
    r'''
           y
           |
        reshape
           |
    x  transpose
     \     /
      matmul
        |
    transpose
        |
     reshape  residual
         \   /
          add
           |
          out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                residual = paddle.static.create_parameter(
                    name="residual", shape=[1], dtype='float32'
                )

                reshape_out = paddle.reshape(y, [0, 0, 0, 0])
                transpose_out = paddle.transpose(reshape_out, perm=[0, 2, 3, 1])
                matmul_out = paddle.matmul(x, transpose_out)
                transpose_out = paddle.transpose(matmul_out, perm=[0, 2, 1, 3])
                reshape_out = paddle.reshape(transpose_out, [0, 0, 25])
                out = paddle.add(reshape_out, residual)
                out = paddle.assign(out)
                self.pass_list = [
                    'reshape_transpose_matmul_fuse_pass',
                    'matmul_transpose_reshape_fuse_pass',
                    'matmul_elementwise_add_fuse_pass',
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "residual": np.random.random(1).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.reshape": 0,
                    "pd_op.transpose": 0,
                    "pd_op.matmul": 0,
                    "pd_op.add": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
