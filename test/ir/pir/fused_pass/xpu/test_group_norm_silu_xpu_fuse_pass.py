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
from paddle.base import core

paddle.enable_static()


class TestGroupNormSiluXpuFusePattern(PassTest):
    r"""
                      X
              Scale   |   Bias
                   \  |  /
                  group norm
                   /  |  \
                  /   |   \
            variance  |   mean
                      |
                     silu
                      |
                    output
    """

    def is_program_valid(self, program):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                channels = 128
                groups = 32
                x = paddle.static.data(
                    name='X', shape=[1, channels, 64, 64], dtype='float32'
                )

                group_norm = paddle.nn.GroupNorm(groups, channels)
                silu = paddle.nn.Silu()

                group_norm_out = group_norm(x)
                out = silu(group_norm_out)
                out = paddle.assign(out)
                self.pass_list = ['group_norm_silu_xpu_fuse_pass']
                self.feeds = {
                    "X": np.random.random((1, channels, 64, 64)).astype(
                        "float32"
                    ),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.group_norm": 0,
                    "pd_op.silu": 0,
                    "pd_op.group_norm_silu_xpu": 1,
                }
                return [main_prog, start_prog]

    def setUp(self):
        if core.is_compiled_with_xpu():
            self.places.append(paddle.XPUPlace(0))
        self.skip_accuracy_verification = True

    def sample_program(self):
        yield self.build_ir_program(), False

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
