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

    def build_ir_program(self, act_layer, residual_first, use_global_stats):
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
                    data_format='NCHW',
                    bias_attr=False,
                )
                bn = paddle.nn.BatchNorm2D(
                    num_features=16,
                    data_format='NCHW',
                    use_global_stats=use_global_stats,
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
        # use_global_stats=True  -> pd_op.batch_norm
        # use_global_stats=False -> pd_op.batch_norm_ (inplace)
        for use_global_stats in (True, False):
            for act_layer in act_layers:
                for residual_first in (True, False):
                    yield (
                        self.build_ir_program(
                            act_layer, residual_first, use_global_stats
                        ),
                        False,
                    )

    def test_check_output(self):
        self.check_pass_correct(atol=2e-3, rtol=2e-3)

    def setUp(self):
        if core.is_compiled_with_xpu():
            self.places.append(paddle.device.XPUPlace(0))
            self.pass_attr_list = [{'conv2d_bn_add_act_xpu_fuse_pass': {}}]
            self.valid_op_map = {
                "pd_op.conv2d_xpu": 1,
                "pd_op.batch_norm_": 0,
                "pd_op.batch_norm": 0,
                "pd_op.add": 0,
                "pd_op.relu": 0,
                "pd_op.swish": 0,
                "pd_op.hardswish": 0,
            }


if __name__ == "__main__":
    unittest.main()
