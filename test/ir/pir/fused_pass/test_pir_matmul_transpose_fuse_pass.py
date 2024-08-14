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

import os
import unittest

import numpy as np
from pass_test import PassTest

import paddle
from paddle.base import core

paddle.enable_static()


class TestMatmulOutTransposeFusePattern(PassTest):
    r"""
    x_var     y_var
       \       /
        \     /
         matmul
           |
       transpose
           |
          out

    x_var   y_var
      \       /
     matmul(tans)
          |
         out

    """

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for x_shape in [[1, 2, 3]]:
            for y_shape in [[1, 3, 2]]:
                for perm in [[0, 2, 1]]:
                    with paddle.pir_utils.IrGuard():
                        main_prog = paddle.static.Program()
                        start_prog = paddle.static.Program()
                        with paddle.static.program_guard(main_prog, start_prog):
                            x = paddle.static.data(
                                name='x', shape=x_shape, dtype='float32'
                            )
                            y = paddle.static.data(
                                name='y', shape=y_shape, dtype='float32'
                            )
                            matmul_out = paddle.matmul(x, y, name='matmul_out')
                            out = paddle.transpose(matmul_out, perm=perm)
                            out = paddle.assign(out)
                            self.pass_attr_list = [
                                {'matmul_transpose_fuse_pass': {}}
                            ]
                            self.feeds = {
                                "x": np.random.random(x_shape).astype(
                                    "float32"
                                ),
                                "y": np.random.random(y_shape).astype(
                                    "float32"
                                ),
                            }
                            self.fetch_list = [out]
                            self.valid_op_map = {
                                "pd_op.matmul": 1,
                                "pd_op.transpose": 0,
                            }
                            yield [main_prog, start_prog], False

    def setUp(self):
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


class TestMatmulYTransposeFusePattern(PassTest):
    r"""
    x_var        y_var
      \           /
       \    transpose
        \     /
        matmul
          |
         out

    x_var   y_var
      \       /
     matmul(tans)
          |
         out
    """

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for x_shape in [[1, 2, 3]]:
            for y_shape in [[1, 2, 3]]:
                for perm in [[0, 2, 1]]:
                    with paddle.pir_utils.IrGuard():
                        main_prog = paddle.static.Program()
                        start_prog = paddle.static.Program()
                        with paddle.static.program_guard(main_prog, start_prog):
                            x = paddle.static.data(
                                name='x', shape=x_shape, dtype='float32'
                            )
                            y = paddle.static.data(
                                name='y', shape=y_shape, dtype='float32'
                            )
                            y_t = paddle.transpose(y, perm)
                            out = paddle.matmul(x, y_t)
                            out = paddle.assign(out)
                            self.pass_attr_list = [
                                {'matmul_transpose_fuse_pass': {}}
                            ]
                            self.feeds = {
                                "x": np.random.random(x_shape).astype(
                                    "float32"
                                ),
                                "y": np.random.random(y_shape).astype(
                                    "float32"
                                ),
                            }
                            self.fetch_list = [out]
                            self.valid_op_map = {
                                "pd_op.matmul": 1,
                                "pd_op.transpose": 0,
                            }
                            yield [main_prog, start_prog], False

    def setUp(self):
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


class TestMatmulXTransposeFusePattern(PassTest):
    r"""
    x_var        y_var
      \           /
    transpose   /
        \     /
        matmul
          |
         out

    x_var   y_var
      \       /
     matmul(tans)
          |
         out
    """

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for x_shape in [[1, 2, 3]]:
            for y_shape in [[1, 2, 3]]:
                for perm in [[0, 2, 1]]:
                    with paddle.pir_utils.IrGuard():
                        main_prog = paddle.static.Program()
                        start_prog = paddle.static.Program()
                        with paddle.static.program_guard(main_prog, start_prog):
                            x = paddle.static.data(
                                name='x', shape=x_shape, dtype='float32'
                            )
                            y = paddle.static.data(
                                name='y', shape=y_shape, dtype='float32'
                            )
                            x_t = paddle.transpose(x, perm)
                            out = paddle.matmul(x_t, y)
                            out = paddle.assign(out)
                            self.pass_attr_list = [
                                {'matmul_transpose_fuse_pass': {}}
                            ]
                            self.feeds = {
                                "x": np.random.random(x_shape).astype(
                                    "float32"
                                ),
                                "y": np.random.random(y_shape).astype(
                                    "float32"
                                ),
                            }
                            self.fetch_list = [out]
                            self.valid_op_map = {
                                "pd_op.matmul": 1,
                                "pd_op.transpose": 0,
                            }
                            yield [main_prog, start_prog], False

    def setUp(self):
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
