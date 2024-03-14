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
import sys
sys.path.append("../")
import numpy as np
from pass_test import PassTest

import paddle

paddle.enable_static()


@unittest.skipIf(
    not paddle.base.core.is_compiled_with_mkldnn(),
    "Test case only for OneDNN pass.",
)
class TestConv2dAddFusePass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
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
                residual_data = paddle.static.data(
                    name="residual_data", shape=[3, 32, 28, 28], dtype="float32"
                )
                out = paddle.add(conv2d(x), residual_data)
                out = paddle.assign(out)
                self.pass_list = ['conv_elementwise_add_mkldnn_fuse_pass']
                self.feeds = {
                    "x": np.random.random((3, 1, 28, 28)).astype("float32"),
                    "residual_data": np.random.random((3, 32, 28, 28)).astype(
                        "float32"
                    ),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_conv2d": 1,
                    "pd_op.conv2d": 0,
                    "pd_op.add": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


@unittest.skipIf(
    not paddle.base.core.is_compiled_with_mkldnn(),
    "Test case only for OneDNN pass.",
)
class TestConv2dAddFusePassAsY(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
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
                residual_data = paddle.static.data(
                    name="residual_data", shape=[3, 32, 28, 28], dtype="float32"
                )
                out = paddle.add(residual_data, conv2d(x))
                out = paddle.assign(out)
                self.pass_list = ['conv_elementwise_add_mkldnn_fuse_pass']
                self.feeds = {
                    "x": np.random.random((3, 1, 28, 28)).astype("float32"),
                    "residual_data": np.random.random((3, 32, 28, 28)).astype(
                        "float32"
                    ),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_conv2d": 1,
                    "pd_op.conv2d": 0,
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
