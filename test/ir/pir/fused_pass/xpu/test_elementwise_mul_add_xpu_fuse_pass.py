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


class TestElementwiseMulAddXpuFusePattern(PassTest):
    r"""
      x         y
       \       /
        \     /
    elementwise_mul    w
            \         /
             \       /
          elementwise_add
                |
                |
              output
    """

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x_shape = [2, 1, 2560]
                x_type = 'float32'
                y_shape = [2, 1, 2560]
                y_type = 'float32'
                w_shape = [2, 1, 2560]
                w_type = 'float32'
                x = paddle.static.data(name='x', shape=x_shape, dtype=x_type)
                y = paddle.static.data(name='y', shape=y_shape, dtype=y_type)
                w = paddle.static.data(name='w', shape=w_shape, dtype=w_type)

                out = paddle.add(paddle.multiply(x, y), w)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'elementwise_mul_add_xpu_fuse_pass': {}}
                ]
                self.feeds = {
                    "x": np.random.random(x_shape).astype("float32"),
                    "y": np.random.random(y_shape).astype("float32"),
                    "w": np.random.random(w_shape).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.multiply": 0,
                    "pd_op.add": 0,
                    "pd_op.addcmul_xpu": 1,
                }

                return [main_prog, start_prog]

    def sample_program(self):
        pir_program = self.build_ir_program()
        yield pir_program, False

    def setUp(self):
        if core.is_compiled_with_xpu():
            self.places.append(paddle.device.XPUPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
