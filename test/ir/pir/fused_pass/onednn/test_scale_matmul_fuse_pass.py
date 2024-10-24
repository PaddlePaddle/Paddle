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


class TestReshapeTranspoeMatmulFusePatternCase1(PassTest):
    r'''
      x
      |
    scale      y
       \      /
        matmul
          |
      matmul_out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[2, 2, 3, 4], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[2, 2, 4, 3], dtype='float32'
                )

                scale_out = paddle.scale(x, scale=0.5)
                matmul_out = paddle.matmul(scale_out, y)
                matmul_out = paddle.assign(matmul_out)
                self.pass_attr_list = [{'scale_matmul_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((2, 2, 3, 4)).astype("float32"),
                    "y": np.random.random((2, 2, 4, 3)).astype("float32"),
                }
                self.fetch_list = [matmul_out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.scale": 0,
                    "pd_op.matmul": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestReshapeTranspoeMatmulFusePatternCase2(PassTest):
    r'''
            y
            |
    x     scale
     \      /
      matmul
        |
    matmul_out
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

                scale_out = paddle.scale(y, scale=0.3)
                matmul_out = paddle.matmul(x, scale_out)
                matmul_out = paddle.assign(matmul_out)
                self.pass_attr_list = [{'scale_matmul_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [matmul_out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.scale": 0,
                    "pd_op.matmul": 0,
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
