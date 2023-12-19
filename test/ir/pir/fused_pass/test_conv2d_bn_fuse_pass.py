# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


@unittest.skipIf(
    not paddle.base.core.is_compiled_with_cuda(),
    "core is not complied with CUDA",
)
class TestConv2dBnPassPattern(PassTest):
    r"""
    x_var   f_var
      \       /
         conv2d
           |
        BatchNorm
    """

    def is_program_valid(self, program=None):
        return True

    def build_ir_progam(self):
        pir_program = None
        with paddle.pir_utils.IrGuard():
            pir_program = paddle.static.Program()
            with paddle.pir.core.program_guard(pir_program):
                x = paddle.static.data(
                    name='x', shape=[3, 1, 28, 28], dtype='float32'
                )
                conv2d = paddle.nn.Conv2D(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                    data_format='NCHW',
                    bias_attr=False,
                )
                bn = paddle.nn.BatchNorm2D(num_features=32, data_format='NCHW')
                out = bn(conv2d(x))

        self.pass_list = ['conv2d_bn_fuse_pass']
        self.feeds = {"x": np.random.random((3, 1, 28, 28)).astype("float32")}
        self.fetch_list = [out]
        self.valid_op_map = {
            "pd_op.conv2d": 1,
            "pd_op.batch_norm": 0,
        }
        return pir_program

    def sample_program(self):
        pir_program = self.build_ir_progam()
        yield pir_program, False

    def setUp(self):
        self.place_runtime = "gpu"

    def test_check_output(self):
        self.check_pass_correct()


class TestConv2dBnPassWtihCpu(TestConv2dBnPassPattern):
    def setUp(self):
        self.place_runtime = "cpu"


@unittest.skipIf(
    not paddle.base.core.is_compiled_with_cuda(),
    "core is not complied with CUDA",
)
class TestBnReplacePattern(PassTest):
    r"""
    BatchNorm
    """

    def is_program_valid(self, program=None):
        return True

    def build_ir_progam(self):
        pir_program = None
        with paddle.pir_utils.IrGuard():
            pir_program = paddle.static.Program()
            with paddle.pir.core.program_guard(pir_program):
                x = paddle.static.data(
                    name='x', shape=[3, 32, 28, 28], dtype='float32'
                )
                bn = paddle.nn.BatchNorm2D(num_features=32, data_format='NCHW')
                out = bn(x)

        self.pass_list = ['conv2d_bn_fuse_pass']
        self.feeds = {"x": np.random.random((3, 32, 28, 28)).astype("float32")}
        self.fetch_list = [out]
        self.valid_op_map = {
            "pd_op.batch_norm_": 1,
            "pd_op.batch_norm": 0,
        }
        return pir_program

    def sample_program(self):
        yield self.build_ir_progam(), False

    def setUp(self):
        self.place_runtime = "gpu"

    def test_check_output(self):
        self.check_pass_correct()


class TestBnReplacePatternWithCpu(TestBnReplacePattern):
    def setUp(self):
        self.place_runtime = "cpu"


if __name__ == "__main__":
    unittest.main()
