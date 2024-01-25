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
from fused_pass.pass_test import PassTest

import paddle
import paddle.nn.functional as F
from paddle.base import core
from paddle.pir.core import create_parameter

paddle.enable_static()


class TestDepthwiseConv2ConvPass(PassTest):
    r""" """

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            for x_shape in [[3, 32, 150, 150]]:
                for conv2d_filter_shape in [[32, 1, 3, 3]]:
                    with paddle.pir.core.program_guard(main_prog, start_prog):
                        x = paddle.static.data(
                            name='x', shape=x_shape, dtype='float32'
                        )
                        initializer = paddle.nn.initializer.Assign(
                            np.random.rand(32, 1, 3, 3)
                        )
                        conv2d_filter = create_parameter(
                            shape=conv2d_filter_shape,
                            dtype='float32',
                            initializer=initializer,
                        )
                        depthwise_conv2d_out = F.conv2d(
                            x, conv2d_filter, groups=32, data_format="NCHW"
                        )
                        bn = paddle.nn.BatchNorm2D(
                            num_features=32,
                            data_format='NCHW',
                            use_global_stats=True,
                        )
                        out = bn(depthwise_conv2d_out)
                        out = paddle.assign(out)
                        self.pass_list = ['depthwise_conv_to_conv_pass']
                        self.feeds = {
                            "x": np.random.random(x_shape).astype("float32"),
                        }
                        self.fetch_list = [out]
                        self.valid_op_map = {
                            "pd_op.depthwise_conv2d": 0,
                            "pd_op.conv2d": 1,
                            "pd_op.add": 0,
                        }
                        yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
