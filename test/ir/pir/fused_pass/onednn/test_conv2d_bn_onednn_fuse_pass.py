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


class TestConv2dBnOneDNNPassPattern(PassTest):
    r"""
    x_var   f_var
      \      /
       conv2d
          |
      BatchNorm
    """

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
                bn = paddle.nn.BatchNorm2D(
                    num_features=32,
                    data_format='NCHW',
                    use_global_stats=True,
                )
                out = bn(conv2d(x))
                out = paddle.assign(out)
                self.pass_attr_list = [{'conv2d_bn_onednn_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((3, 1, 28, 28)).astype("float32")
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_conv2d": 1,
                    "pd_op.batch_norm_": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        pir_program = self.build_ir_program()
        yield pir_program, False

    def test_check_output(self):
        self.check_pass_correct()

    def setUp(self):
        self.places.append(paddle.CPUPlace())


class TestConv2dBiasBnOneDNNPassPattern(PassTest):
    r"""
    x_var   f_var
      \      /
       conv2d  add_y
          \     /
            add
             |
         BatchNorm
             |
    """

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
                bias_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                y = paddle.static.create_parameter(
                    shape=[1],
                    dtype='float32',
                    attr=bias_attr,
                    is_bias=False,
                )
                conv2d = paddle.nn.Conv2D(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                    data_format='NCHW',
                    bias_attr=False,
                )
                bn = paddle.nn.BatchNorm2D(
                    num_features=32,
                    data_format='NCHW',
                    use_global_stats=True,
                )
                add_out = paddle.add(conv2d(x), y)
                out = bn(add_out)
                out = paddle.assign(out)
                self.pass_attr_list = [{'conv2d_bias_bn_onednn_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((3, 1, 28, 28)).astype("float32"),
                    "y": np.random.random(1).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_conv2d": 1,
                    "pd_op.batch_norm_": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        pir_program = self.build_ir_program()
        yield pir_program, False

    def test_check_output(self):
        self.check_pass_correct()

    def setUp(self):
        self.places.append(paddle.CPUPlace())


class TestConv2dBiasBnOneDNNPassPatternCase2(PassTest):
    r"""
    x_var   f_var
      \      /
       conv2d  add_y
          \     /
            add
             |
         BatchNorm
             |
    """

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
                bias_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                y = paddle.static.create_parameter(
                    shape=[1, 32, 1, 1],
                    dtype='float32',
                    attr=bias_attr,
                    is_bias=False,
                )
                conv2d = paddle.nn.Conv2D(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                    data_format='NCHW',
                    bias_attr=False,
                )
                bn = paddle.nn.BatchNorm2D(
                    num_features=32,
                    data_format='NCHW',
                    use_global_stats=True,
                )
                add_out = paddle.add(conv2d(x), y)
                out = bn(add_out)
                out = paddle.assign(out)
                self.pass_attr_list = [{'conv2d_bias_bn_onednn_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((3, 1, 28, 28)).astype("float32"),
                    "y": np.random.random((1, 32, 1, 1)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_conv2d": 1,
                    "pd_op.batch_norm_": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        pir_program = self.build_ir_program()
        yield pir_program, False

    def test_check_output(self):
        self.check_pass_correct()

    def setUp(self):
        self.places.append(paddle.CPUPlace())


class TestConv2dBiasBnOneDNNPassPatternCase3(PassTest):
    r"""
    x_var   f_var
      \      /
       conv2d  add_y
          \     /
            add
             |
         BatchNorm
             |
    """

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[3, 28, 28, 1], dtype='float32'
                )
                bias_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                y = paddle.static.create_parameter(
                    shape=[1, 1, 1, 32],
                    dtype='float32',
                    attr=bias_attr,
                    is_bias=False,
                )
                conv2d = paddle.nn.Conv2D(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                    data_format='NHWC',
                    bias_attr=False,
                )
                bn = paddle.nn.BatchNorm2D(
                    num_features=32,
                    data_format='NHWC',
                    use_global_stats=True,
                )
                add_out = paddle.add(conv2d(x), y)
                out = bn(add_out)
                out = paddle.assign(out)
                self.pass_attr_list = [{'conv2d_bias_bn_onednn_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((3, 28, 28, 1)).astype("float32"),
                    "y": np.random.random((1, 1, 1, 32)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_conv2d": 1,
                    "pd_op.batch_norm_": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        pir_program = self.build_ir_program()
        yield pir_program, False

    def test_check_output(self):
        self.check_pass_correct()

    def setUp(self):
        self.places.append(paddle.CPUPlace())


if __name__ == "__main__":
    unittest.main()
