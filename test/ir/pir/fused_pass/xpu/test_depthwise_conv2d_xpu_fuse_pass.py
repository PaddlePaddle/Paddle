# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

paddle.enable_static()


class TestDepthwiseConv2dXpuFusePattern(PassTest):
    r"""
      x_var
        |
    depthwise_conv2d   ==>   conv2d_xpu (act=LINEAR, no_bias, no_branch)
        |
       out
    """

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[2, 16, 16, 16], dtype='float32'
                )
                dw_conv = paddle.nn.Conv2D(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=3,
                    padding=1,
                    groups=16,
                    data_format='NCHW',
                    bias_attr=False,
                )
                out = paddle.assign(dw_conv(x))
                self.feeds = {
                    "x": np.random.random((2, 16, 16, 16)).astype("float32")
                }
                self.fetch_list = [out]
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def test_check_output(self):
        self.check_pass_correct(atol=2e-3, rtol=2e-3)

    def setUp(self):
        if core.is_compiled_with_xpu():
            self.places.append(paddle.device.XPUPlace(0))
            self.pass_attr_list = [{'depthwise_conv2d_xpu_fuse_pass': {}}]
            self.valid_op_map = {
                "pd_op.conv2d_xpu": 1,
                "pd_op.depthwise_conv2d": 0,
            }


if __name__ == "__main__":
    unittest.main()
