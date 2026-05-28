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
            self.pass_attr_list = [{'conv2d_xpu_fuse_pass': {}}]
            self.valid_op_map = {
                "pd_op.conv2d_xpu": 1,
                "pd_op.depthwise_conv2d": 0,
            }


class TestConv2dBnXpuFusePattern(PassTest):
    r"""
    x_var   f_var
      \       /
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
                self.feeds = {
                    "x": np.random.random((3, 1, 28, 28)).astype("float32")
                }
                self.fetch_list = [out]
                return [main_prog, start_prog]

    def sample_program(self):
        pir_program = self.build_ir_program()
        yield pir_program, False

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)

    def setUp(self):
        if core.is_compiled_with_xpu():
            self.places.append(paddle.device.XPUPlace(0))
            self.pass_attr_list = [{'conv2d_xpu_fuse_pass': {}}]
            self.valid_op_map = {
                "pd_op.conv2d_xpu": 1,
                "pd_op.batch_norm": 0,
            }


class TestConv2dBnActXpuFusePattern(PassTest):
    r"""
    x_var   f_var
      \       /
         conv2d (or depthwise_conv2d)
           |
        BatchNorm
           |
          Act
           |
          out
    """

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self, in_channels, groups, act_layer):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x',
                    shape=[2, in_channels, 16, 16],
                    dtype='float32',
                )
                conv2d = paddle.nn.Conv2D(
                    in_channels=in_channels,
                    out_channels=in_channels if groups > 1 else 32,
                    kernel_size=3,
                    padding=1,
                    groups=groups,
                    data_format='NCHW',
                    bias_attr=False,
                )
                bn = paddle.nn.BatchNorm2D(
                    num_features=conv2d._out_channels,
                    data_format='NCHW',
                    use_global_stats=True,
                )
                out = act_layer(bn(conv2d(x)))
                out = paddle.assign(out)
                self.feeds = {
                    "x": np.random.random((2, in_channels, 16, 16)).astype(
                        "float32"
                    ),
                }
                self.fetch_list = [out]
                return [main_prog, start_prog]

    def sample_program(self):
        # (in_channels, groups) covers both regular and depthwise convolutions.
        # Only use_global_stats=True is exercised here: in train mode (=False)
        # the reference run uses batch statistics while the fusion folds the
        # running statistics, so the numerical check would not be meaningful.
        conv_cfgs = [(8, 1), (8, 8)]
        act_layers = [
            paddle.nn.ReLU(),
            paddle.nn.Swish(),
            paddle.nn.Hardswish(),
        ]
        for in_channels, groups in conv_cfgs:
            for act_layer in act_layers:
                yield (
                    self.build_ir_program(in_channels, groups, act_layer),
                    False,
                )

    def test_check_output(self):
        self.check_pass_correct(atol=2e-3, rtol=2e-3)

    def setUp(self):
        if core.is_compiled_with_xpu():
            self.places.append(paddle.device.XPUPlace(0))
            self.pass_attr_list = [{'conv2d_xpu_fuse_pass': {}}]
            self.valid_op_map = {
                "pd_op.conv2d_xpu": 1,
                "pd_op.batch_norm_": 0,
                "pd_op.batch_norm": 0,
                "pd_op.relu": 0,
                "pd_op.swish": 0,
                "pd_op.hardswish": 0,
            }


class TestConv2dBnAddActXpuFusePattern(PassTest):
    r"""
    x_var          branch_var
      |               |
    conv2d            |
      |               |
    BatchNorm         |
      \              /
          add (residual)
            |
           Act
            |
           out
    """

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self, act_layer, residual_first, groups):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[2, 16, 16, 16], dtype='float32'
                )
                branch = paddle.static.data(
                    name='branch', shape=[2, 16, 16, 16], dtype='float32'
                )
                conv2d = paddle.nn.Conv2D(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=3,
                    padding=1,
                    groups=groups,
                    data_format='NCHW',
                    bias_attr=False,
                )
                bn = paddle.nn.BatchNorm2D(
                    num_features=16,
                    data_format='NCHW',
                    use_global_stats=True,
                )
                bn_out = bn(conv2d(x))
                if residual_first:
                    add_out = paddle.add(branch, bn_out)
                else:
                    add_out = paddle.add(bn_out, branch)
                out = act_layer(add_out)
                out = paddle.assign(out)
                self.feeds = {
                    "x": np.random.random((2, 16, 16, 16)).astype("float32"),
                    "branch": np.random.random((2, 16, 16, 16)).astype(
                        "float32"
                    ),
                }
                self.fetch_list = [out]
                return [main_prog, start_prog]

    def sample_program(self):
        act_layers = [
            paddle.nn.ReLU(),
            paddle.nn.Swish(),
            paddle.nn.Hardswish(),
        ]
        # groups=1: regular conv2d; groups=16: depthwise_conv2d.
        # Only use_global_stats=True is exercised: in train mode (=False)
        # the reference run uses batch statistics while the fusion folds the
        # running statistics, so the numerical check would not be meaningful.
        for groups in (1, 16):
            for act_layer in act_layers:
                for residual_first in (True, False):
                    yield (
                        self.build_ir_program(
                            act_layer, residual_first, groups
                        ),
                        False,
                    )

    def test_check_output(self):
        self.check_pass_correct(atol=2e-3, rtol=2e-3)

    def setUp(self):
        if core.is_compiled_with_xpu():
            self.places.append(paddle.device.XPUPlace(0))
            self.pass_attr_list = [{'conv2d_xpu_fuse_pass': {}}]
            self.valid_op_map = {
                "pd_op.conv2d_xpu": 1,
                "pd_op.batch_norm_": 0,
                "pd_op.batch_norm": 0,
                # The residual `add(branch, bn_out)` is fused into conv2d_xpu's
                # branch input. The single remaining `pd_op.add` is the
                # `bn_var + epsilon` op emitted by the BN-fold subgraph; it
                # operates on persistable constants and would be eliminated by
                # a subsequent constant-folding pass.
                "pd_op.add": 1,
                "pd_op.relu": 0,
                "pd_op.swish": 0,
                "pd_op.hardswish": 0,
            }


if __name__ == "__main__":
    unittest.main()
