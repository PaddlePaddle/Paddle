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


class TestConv2dConcatReluFusePass(PassTest):
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
                conv2d = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                )
                act_op = paddle.nn.ReLU()
                concat_out = paddle.concat([conv2d(x)])

                out = act_op(concat_out)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'conv_concat_activation_onednn_fuse_pass': {}}
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.relu": 0,
                    "pd_op.conv2d": 0,
                    "pd_op.concat": 1,
                    "onednn_op.fused_conv2d": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestConv2dConcat3ReluFusePass(PassTest):
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
                conv2d = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                )
                x1 = paddle.static.data(
                    name='x1', shape=[5, 5, 5, 5], dtype='float32'
                )
                conv2d1 = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                )
                x2 = paddle.static.data(
                    name='x2', shape=[5, 5, 5, 5], dtype='float32'
                )
                conv2d2 = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                )

                act_op = paddle.nn.ReLU()

                concat_out = paddle.concat(
                    [conv2d(x), conv2d1(x1), conv2d2(x2)]
                )

                out = act_op(concat_out)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'conv_concat_activation_onednn_fuse_pass': {}}
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "x1": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "x2": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.relu": 0,
                    "pd_op.conv2d": 0,
                    "pd_op.concat": 1,
                    "onednn_op.fused_conv2d": 3,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestConv2dConcat3GELUFusePass(PassTest):
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
                conv2d = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                )
                x1 = paddle.static.data(
                    name='x1', shape=[5, 5, 5, 5], dtype='float32'
                )
                conv2d1 = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                )
                x2 = paddle.static.data(
                    name='x2', shape=[5, 5, 5, 5], dtype='float32'
                )
                conv2d2 = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                )

                act_op = paddle.nn.GELU()
                concat_out = paddle.concat(
                    [conv2d(x), conv2d1(x1), conv2d2(x2)]
                )

                out = act_op(concat_out)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'conv_concat_activation_onednn_fuse_pass': {}}
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "x1": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "x2": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.gelu": 0,
                    "pd_op.conv2d": 0,
                    "pd_op.concat": 1,
                    "onednn_op.fused_conv2d": 3,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestConv2dConcat3HardsigmoidFusePass(PassTest):
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
                conv2d = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                )
                x1 = paddle.static.data(
                    name='x1', shape=[5, 5, 5, 5], dtype='float32'
                )
                conv2d1 = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                )
                x2 = paddle.static.data(
                    name='x2', shape=[5, 5, 5, 5], dtype='float32'
                )
                conv2d2 = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                )

                act_op = paddle.nn.Hardsigmoid()
                concat_out = paddle.concat(
                    [conv2d(x), conv2d1(x1), conv2d2(x2)]
                )

                out = act_op(concat_out)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'conv_concat_activation_onednn_fuse_pass': {}}
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "x1": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "x2": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.hardsigmoid": 0,
                    "pd_op.conv2d": 0,
                    "pd_op.concat": 1,
                    "onednn_op.fused_conv2d": 3,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestConv2dConcat3ClipFusePass(PassTest):
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
                conv2d = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                )
                x1 = paddle.static.data(
                    name='x1', shape=[5, 5, 5, 5], dtype='float32'
                )
                conv2d1 = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                )

                concat_out = paddle.concat([conv2d(x), conv2d1(x1)])

                out = paddle.clip(concat_out, min=-15, max=15)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'conv_concat_activation_onednn_fuse_pass': {}}
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "x1": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.clip": 0,
                    "pd_op.conv2d": 0,
                    "pd_op.concat": 1,
                    "onednn_op.fused_conv2d": 2,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestConv2dConcat6ReluFusePass(PassTest):
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
                conv2d = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                )
                x1 = paddle.static.data(
                    name='x1', shape=[5, 5, 5, 5], dtype='float32'
                )
                conv2d1 = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                )
                x2 = paddle.static.data(
                    name='x2', shape=[5, 5, 5, 5], dtype='float32'
                )
                conv2d2 = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                )
                x3 = paddle.static.data(
                    name='x3', shape=[5, 5, 5, 5], dtype='float32'
                )
                conv2d3 = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                )
                x4 = paddle.static.data(
                    name='x4', shape=[5, 5, 5, 5], dtype='float32'
                )
                conv2d4 = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                )
                x5 = paddle.static.data(
                    name='x5', shape=[5, 5, 5, 5], dtype='float32'
                )
                conv2d5 = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                )

                act_op = paddle.nn.GELU()
                concat_out = paddle.concat(
                    [
                        conv2d(x),
                        conv2d1(x1),
                        conv2d2(x2),
                        conv2d3(x3),
                        conv2d4(x4),
                        conv2d5(x5),
                    ]
                )

                out = act_op(concat_out)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'conv_concat_activation_onednn_fuse_pass': {}}
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "x1": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "x2": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "x3": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "x4": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "x5": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.relu": 0,
                    "pd_op.conv2d": 0,
                    "pd_op.concat": 1,
                    "onednn_op.fused_conv2d": 6,
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
