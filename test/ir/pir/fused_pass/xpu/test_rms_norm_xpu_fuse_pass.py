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
from paddle.pir.core import create_parameter

paddle.enable_static()


class TestRmsNormXpuFusePattern(PassTest):
    r"""
     x                   x       w
     |                   |       |
    pow                  |       |
     |                   |       |
    mean     epilson     |       |
       \     /           |       |
        rsqrt            |       |
          |              |       |
            \          /         |
              multiply           |
                 |               |
                    \          /
                      multiply
    """

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x_shape = [2, 160, 40, 64]
                x_type = 'float32'
                w_shape = [64]
                w_type = 'float32'
                x = paddle.static.data(name='x', shape=x_shape, dtype=x_type)
                w = create_parameter(
                    name="w",
                    shape=w_shape,
                    dtype=w_type,
                    initializer=paddle.nn.initializer.Assign(
                        np.random.random(w_shape).astype(w_type)
                    ),
                )
                variance = x.pow(2).mean(-1, keepdim=True)
                x = paddle.rsqrt(variance + 1e-6) * x
                out = x * w
                out = paddle.assign(out)
                self.pass_attr_list = [{'rms_norm_xpu_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random(x_shape).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.pow": 0,
                    "pd_op.mean": 0,
                    "pd_op.full": 0,
                    "pd_op.scale": 0,
                    "pd_op.rsqrt": 0,
                    "pd_op.multiply": 0,
                    "pd_op.rms_norm": 1,
                }

                return [main_prog, start_prog]

    def sample_program(self):
        pir_program = self.build_ir_program()
        yield pir_program, False

    def setUp(self):
        if core.is_compiled_with_xpu():
            self.places.append(paddle.device.XPUPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)


# 因为无法使用numpy构造bfp16类型得数据，所以这里仅用fp16来测试半精度的case
class TestRmsNorm_FP16_XpuFusePattern(PassTest):
    r"""
              x                  w
     _ _ _ _ _| _ _ _ _ _        |
     |                  |        |
    cast               cast      |
     |                   |       |
     |                   |       |
    pow                  |       |
     |                   |       |
    mean     epilson     |       |
       \     /           |       |
        rsqrt            |       |
          |              |       |
            \          /         |
              multiply           |
                 |               |
                cast             |
                    \          /
                      multiply
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
                x_shape = [2, 160, 40, 64]
                x_type = 'float16'
                w_shape = [64]
                w_type = 'float16'
                x = paddle.static.data(name='x', shape=x_shape, dtype=x_type)
                x_fp32_1 = paddle.cast(x, 'float32')
                x_fp32_2 = paddle.cast(x, 'float32')
                w = create_parameter(
                    name="w",
                    shape=w_shape,
                    dtype=w_type,
                    initializer=paddle.nn.initializer.Assign(
                        np.random.random(w_shape).astype(w_type)
                    ),
                )
                variance = x_fp32_1.pow(2).mean(-1, keepdim=True)
                x_fp32_1 = paddle.rsqrt(variance + 1e-6) * x_fp32_2
                x_float16 = paddle.cast(x_fp32_1, 'float16')
                out = x_float16 * w
                out = paddle.assign(out)
                self.pass_attr_list = [{'rms_norm_xpu_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random(x_shape).astype("float16"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.pow": 0,
                    "pd_op.mean": 0,
                    "pd_op.full": 0,
                    "pd_op.scale": 0,
                    "pd_op.rsqrt": 0,
                    "pd_op.multiply": 0,
                    "pd_op.rms_norm": 1,
                }

                return [main_prog, start_prog]

    def sample_program(self):
        pir_program = self.build_ir_program()
        yield pir_program, False

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)

    def setUp(self):
        if core.is_compiled_with_xpu():
            self.places.append(paddle.device.XPUPlace(0))


if __name__ == "__main__":
    unittest.main()
