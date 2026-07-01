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


class TestConcatConvXpuFusePattern(PassTest):
    r"""
    Eliminate the channel-axis concat that feeds a 1x1 conv2d_xpu.

    Input graph (after conv2d_xpu_fuse_pass has fused each branch conv+bn+act
    and the transition conv+bn+act into conv2d_xpu):

        x0 -> conv2d(3x3) -> bn -> relu -+
        x1 -> conv2d(3x3) -> bn -> relu -+--> concat(axis=1) -> conv2d(1x1)+bn+relu -> out
        ...

    Expected after concat_conv_xpu_fuse_pass:
        - pd_op.concat is eliminated (count == 0)
        - the single conv2d_xpu on the concat output is replaced by N chained
          conv2d_xpu ops that branch-accumulate; concat data movement is gone.
    """

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self, n_branches, in_channels, branch_channels):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                branch_outs = []
                self.feeds = {}
                for i in range(n_branches):
                    xi = paddle.static.data(
                        name=f'x{i}',
                        shape=[1, in_channels, 8, 8],
                        dtype='float32',
                    )
                    self.feeds[f"x{i}"] = np.random.random(
                        (1, in_channels, 8, 8)
                    ).astype("float32")
                    conv = paddle.nn.Conv2D(
                        in_channels=in_channels,
                        out_channels=branch_channels,
                        kernel_size=3,
                        padding=1,
                        data_format='NCHW',
                        bias_attr=False,
                    )
                    bn = paddle.nn.BatchNorm2D(
                        num_features=branch_channels,
                        data_format='NCHW',
                        use_global_stats=True,
                    )
                    out = paddle.nn.functional.relu(bn(conv(xi)))
                    branch_outs.append(out)

                concat_out = paddle.concat(branch_outs, axis=1)
                total_c = n_branches * branch_channels
                transition = paddle.nn.Conv2D(
                    in_channels=total_c,
                    out_channels=16,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    data_format='NCHW',
                    bias_attr=False,
                )
                transition_bn = paddle.nn.BatchNorm2D(
                    num_features=16,
                    data_format='NCHW',
                    use_global_stats=True,
                )
                out = paddle.nn.functional.relu(
                    transition_bn(transition(concat_out))
                )
                out = paddle.assign(out)
                self.fetch_list = [out]
                return [main_prog, start_prog]

    def sample_program(self):
        # 2..4 branches, matching the det/rec channel-concat shapes.
        for n_branches in (2, 3, 4):
            yield (
                self.build_ir_program(
                    n_branches, in_channels=8, branch_channels=8
                ),
                False,
            )

    def test_check_output(self):
        self.check_pass_correct(atol=2e-3, rtol=2e-3)

    def setUp(self):
        if core.is_compiled_with_xpu():
            self.places.append(paddle.device.XPUPlace(0))
            # conv2d_xpu_fuse_pass must run first so the branch convs and the
            # transition conv are already conv2d_xpu; then
            # concat_conv_xpu_fuse_pass eliminates the concat.
            self.pass_attr_list = [
                {'conv2d_xpu_fuse_pass': {}},
                {'concat_conv_xpu_fuse_pass': {}},
            ]
            self.valid_op_map = {
                "pd_op.concat": 0,
            }


if __name__ == "__main__":
    unittest.main()
