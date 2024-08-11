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


class TestAddActivationXpuFusePattern(PassTest):
    r"""
    x_var   y_var
    \      /
       add
        |
     add_var
        |
       act
        |
      out_var
    """

    def is_program_valid(self, program):
        return True

    def build_ir_program(self, act_fun):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[3, 64, 28, 28], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[3, 64, 28, 28], dtype='float32'
                )
                add_out = paddle.add(x, y)
                out = act_fun(add_out)

                out = paddle.assign(out)
                self.pass_attr_list = [{'add_activation_xpu_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((3, 64, 28, 28)).astype("float32"),
                    "y": np.random.random((3, 64, 28, 28)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.add": 0,
                    "pd_op.relu": 0,
                    "pd_op.add_act_xpu": 1,
                }
                return [main_prog, start_prog]

    def setUp(self):
        if core.is_compiled_with_xpu():
            self.places.append(paddle.XPUPlace(0))
        self.skip_accuracy_verification = True

    def sample_program(self):
        act_funs = [paddle.nn.GELU(), paddle.nn.ReLU()]
        for act_fun in act_funs:
            yield self.build_ir_program(act_fun), False

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
