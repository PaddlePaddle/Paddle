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


@unittest.skipIf(
    not paddle.base.core.is_compiled_with_mkldnn(),
    "Test case only for OneDNN pass.",
)
class TestConv3dAddFusePass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5, 5], dtype='float32'
                )
                bias_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                bias = paddle.static.create_parameter(
                    shape=[1], dtype='float32', attr=bias_attr, is_bias=False
                )
                w_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                conv3d = paddle.nn.Conv3D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1, 1],
                    groups=1,
                    stride=[1, 1, 1],
                    padding=[1, 1, 1, 1, 1, 1],
                    dilation=[1, 1, 1],
                    data_format='NCDHW',
                    bias_attr=False,
                    weight_attr=w_attr,
                )

                out = paddle.add(conv3d(x), bias)
                out = paddle.assign(out)
                self.pass_list = ['conv3d_bias_fuse_pass']
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5, 5)).astype("float32"),
                    "bias": np.random.random(1).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_conv3d": 1,
                    "pd_op.conv3d": 0,
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
class TestConv3dAddFusePassWithAddParam(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5, 5], dtype='float32'
                )
                bias_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                bias = paddle.static.create_parameter(
                    shape=[1], dtype='float32', attr=bias_attr, is_bias=False
                )
                w_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                conv3d = paddle.nn.Conv3D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1, 1],
                    groups=1,
                    stride=[1, 1, 1],
                    padding=[1, 1, 1, 1, 1, 1],
                    dilation=[1, 1, 1],
                    data_format='NCDHW',
                    bias_attr=False,
                    weight_attr=w_attr,
                )
                add_out = paddle.add(conv3d(x), bias)
                other_param_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                other_param = paddle.static.create_parameter(
                    shape=[1], dtype='float32', attr=bias_attr, is_bias=False
                )
                out = paddle.add(add_out, other_param)
                out = paddle.assign(out)
                self.pass_list = ['conv3d_bias_fuse_pass']
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5, 5)).astype("float32"),
                    "bias": np.random.random(1).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_conv3d": 1,
                    "pd_op.conv3d": 0,
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
