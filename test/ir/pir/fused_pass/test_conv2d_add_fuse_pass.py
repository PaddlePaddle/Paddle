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
class TestConv2dAddFusePass(PassTest):
    r"""
    x_var   f_var
      \       /
         conv2d
           |
          add
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
                y = paddle.static.data(
                    name="y", shape=[3, 32, 28, 28], dtype="float32"
                )
                conv2d = paddle.nn.Conv2D(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                    data_format='NCHW',
                    bias_attr=False,
                )
                out = paddle.add(conv2d(x), y)

        self.pass_list = ['conv2d_add_fuse_pass']
        self.feeds = {
            "x": np.random.random((3, 1, 28, 28)).astype("float32"),
            "y": np.random.random((3, 32, 28, 28)).astype("float32"),
        }
        self.fetch_list = [out]
        self.valid_op_map = {
            "pd_op.fused_conv2d_add_act": 1,
            "pd_op.conv2d": 0,
            "pd_op.add": 0,
        }
        return pir_program

    def sample_program(self):
        yield self.build_ir_progam(), False

    def setUp(self):
        self.place_runtime = "gpu"

    def test_check_output(self):
        self.check_pass_correct()


class TestConv2dAddFusePassWtihCpu(TestConv2dAddFusePass):
    def setUp(self):
        self.place_runtime = "cpu"


if __name__ == "__main__":
    unittest.main()
