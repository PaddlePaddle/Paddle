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


class TestMatmulAddFusePattern(PassTest):
    r'''
    x     y
     \   /
     matmul
       |
     reshape (bias)
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
                reshape_x = paddle.reshape(x, [5, 125])
                reshape_y = paddle.reshape(y, [125, 5])
                residual = paddle.static.data(
                    name="residual", shape=[5], dtype='float32'
                )
                matmul_out = paddle.matmul(reshape_x, reshape_y)
                out = paddle.add(paddle.reshape(matmul_out, [5, 5]), residual)
                out = paddle.assign(out)
                self.pass_attr_list = [{'matmul_reshape_add_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "residual": np.random.random(5).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
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


class TestMatmulAddFusePattern2(PassTest):
    r'''
    x     y
     \   /
     matmul
       |
     reshape (bias)
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
                # This case, matmul(x,y) shape is 4 dims, can not use matmul+add ->fc fusion
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                residual = paddle.static.data(
                    name="residual", shape=[5], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.add(
                    paddle.reshape(matmul_out, [5, 5, 5, 5]), residual
                )
                out = paddle.assign(out)
                self.pass_attr_list = [{'matmul_reshape_add_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "residual": np.random.random(5).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.matmul": 1,
                    "pd_op.add": 1,
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
