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


class TestFcXpuFuseAddPattern(PassTest):
    r"""
    x        w
     \      /
       fc
        |
       add
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
                    name='x', shape=[2, 128, 64], dtype='float32'
                )
                w = paddle.static.data(
                    name='w', shape=[64, 64], dtype='float32'
                )
                y = paddle.static.data(name='bias', shape=[64], dtype='float32')
                fc_out = paddle.add(paddle.matmul(x, w), y)
                out = paddle.assign(fc_out)
                self.feeds = {
                    "x": np.random.random((2, 128, 64)).astype("float32"),
                    "w": np.random.random((64, 64)).astype("float32"),
                    "bias": np.random.random(64).astype("float32"),
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
            self.pass_attr_list = [{'fc_xpu_fuse_pass': {}}]
            self.valid_op_map = {
                "pd_op.fc_xpu": 1,
                "pd_op.batch_norm": 0,
                "pd_op.relu": 0,
            }


if __name__ == "__main__":
    unittest.main()
