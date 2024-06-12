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
     reshape
        |
    transpose    y
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

                reshape_out = paddle.reshape(x, shape=[2, 2, 3, 4])
                transpose_out = paddle.transpose(reshape_out, perm=[1, 0, 2, 3])
                matmul_out = paddle.matmul(transpose_out, y)
                matmul_out = paddle.assign(matmul_out)
                self.pass_attr_list = [
                    {'reshape_transpose_matmul_fuse_pass': {}}
                ]
                self.feeds = {
                    "x": np.random.random((2, 2, 3, 4)).astype("float32"),
                    "y": np.random.random((2, 2, 4, 3)).astype("float32"),
                }
                self.fetch_list = [matmul_out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.reshape": 0,
                    "pd_op.transpose": 0,
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
         reshape
            |
    x   transpose
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

                reshape_out = paddle.reshape(y, [0, 0, 0, 0])
                transpose_out = paddle.transpose(reshape_out, perm=[0, 2, 3, 1])
                matmul_out = paddle.matmul(x, transpose_out)
                matmul_out = paddle.assign(matmul_out)
                self.pass_attr_list = [
                    {'reshape_transpose_matmul_fuse_pass': {}}
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [matmul_out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.reshape": 0,
                    "pd_op.transpose": 0,
                    "pd_op.matmul": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestReshapeTranspoeMatmulFusePatternCase3(PassTest):
    r'''
        x        y
        |        |
     reshape  reshape
        |        |
    transpose transpose
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

                reshape_x = paddle.reshape(x, [0, 0, 0, 0])
                reshape_y = paddle.reshape(y, [0, 0, 0, 0])
                transpose_x = paddle.transpose(reshape_x, perm=[0, 2, 3, 1])
                transpose_y = paddle.transpose(reshape_y, perm=[0, 2, 3, 1])
                matmul_out = paddle.matmul(transpose_x, transpose_y)
                matmul_out = paddle.assign(matmul_out)
                self.pass_attr_list = [
                    {'reshape_transpose_matmul_fuse_pass': {}},
                    {'reshape_transpose_matmul_fuse_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [matmul_out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.reshape": 0,
                    "pd_op.transpose": 0,
                    "pd_op.matmul": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestReshapeTransposeMatmulAddFusePattern(PassTest):
    r'''
           x
           |
        reshape
           |
    y  transpose
     \    /
     matmul  resdual
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
                residual = paddle.static.data(
                    name="residual", shape=[1], dtype='float32'
                )

                reshape_out = paddle.reshape(y, [0, 0, 0, 0])
                transpose_out = paddle.transpose(reshape_out, perm=[0, 2, 3, 1])
                matmul_out = paddle.matmul(x, transpose_out)
                out = paddle.add(matmul_out, residual)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'reshape_transpose_matmul_fuse_pass': {}},
                    {'matmul_elementwise_add_fuse_pass': {}},
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
