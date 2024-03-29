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


class TestRmsNormFusePattern(PassTest):
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

    def sample_program(self):
        for x_shape in [[1, 1, 4096]]:
            for w_shape in [[4096]]:
                for w_type in ['float32']:
                    for epilson in [1e-6]:
                        with paddle.pir_utils.IrGuard():
                            start_prog = paddle.static.Program()
                            main_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(
                                main_prog, start_prog
                            ):
                                x = paddle.static.data(
                                    name='x', shape=x_shape, dtype='float32'
                                )
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
                                self.pass_list = ['add_norm_fuse_pass']
                                self.feeds = {
                                    "x": np.random.random(x_shape).astype(
                                        "float32"
                                    ),
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

                                yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


class TestRmsNormFusePattern_FP16(TestRmsNormFusePattern):
    r"""
                x                w
                |                |
               cast              |
      _ _ _ _ _ | _ _ _ _        |
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
    """

    def sample_program(self):
        for x_shape in [[1, 1, 4096]]:
            for w_shape in [[4096]]:
                for w_type in ['float16']:
                    for epilson in [1e-6]:
                        paddle.set_default_dtype(w_type)
                        with paddle.pir_utils.IrGuard():
                            start_prog = paddle.static.Program()
                            main_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(
                                main_prog, start_prog
                            ):
                                x = paddle.static.data(
                                    name='x',
                                    shape=x_shape,
                                    dtype=paddle.get_default_dtype(),
                                )
                                x_1 = paddle.cast(x, 'float32')
                                w = create_parameter(
                                    name="w",
                                    shape=w_shape,
                                    dtype=w_type,
                                    initializer=paddle.nn.initializer.Assign(
                                        np.random.random(w_shape).astype(
                                            paddle.get_default_dtype()
                                        )
                                    ),
                                )
                                variance = x_1.pow(2).mean(-1, keepdim=True)
                                x_1 = paddle.rsqrt(variance + 1e-6) * x_1
                                x_2 = paddle.cast(
                                    x_1, paddle.get_default_dtype()
                                )
                                out = x_2 * w
                                out = paddle.assign(out)
                                self.pass_list = ['add_norm_fuse_pass']
                                self.feeds = {
                                    "x": np.random.random(x_shape).astype(
                                        paddle.get_default_dtype()
                                    ),
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

                                yield [main_prog, start_prog], False

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)


class TestAddRmsNormFusePattern(TestRmsNormFusePattern):
    r"""
        x         residual       w
        |           |
             add
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

    def sample_program(self):
        for x_shape in [[1, 1, 4096]]:
            for w_shape in [[4096]]:
                for w_type in ['float32']:
                    for epilson in [1e-6]:
                        with paddle.pir_utils.IrGuard():
                            start_prog = paddle.static.Program()
                            main_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(
                                main_prog, start_prog
                            ):
                                residual = paddle.static.data(
                                    name='residual',
                                    shape=x_shape,
                                    dtype='float32',
                                )
                                x = paddle.static.data(
                                    name='x', shape=x_shape, dtype='float32'
                                )
                                w = create_parameter(
                                    name="w",
                                    shape=w_shape,
                                    dtype=w_type,
                                    initializer=paddle.nn.initializer.Assign(
                                        np.random.random(w_shape).astype(w_type)
                                    ),
                                )
                                add_out = paddle.add(residual, x)
                                variance = add_out.pow(2).mean(-1, keepdim=True)
                                add_out = (
                                    paddle.rsqrt(variance + 1e-6) * add_out
                                )
                                out = add_out * w
                                out = paddle.assign(out)
                                self.pass_list = ['add_norm_fuse_pass']
                                self.feeds = {
                                    "x": np.random.random(x_shape).astype(
                                        "float32"
                                    ),
                                    "residual": np.random.random(
                                        x_shape
                                    ).astype("float32"),
                                }
                                self.fetch_list = [out]
                                self.valid_op_map = {
                                    "pd_op.add": 0,
                                    "pd_op.pow": 0,
                                    "pd_op.mean": 0,
                                    "pd_op.full": 0,
                                    "pd_op.scale": 0,
                                    "pd_op.rsqrt": 0,
                                    "pd_op.multiply": 0,
                                    "pd_op.rms_norm": 1,
                                }

                                yield [main_prog, start_prog], False


class TestAddLayerNormFusePattern(TestRmsNormFusePattern):
    r"""
    x         residual
    |           |
         add
          |
      layer_norm

    """

    def sample_program(self):
        for x_shape in [[1, 1, 4096]]:
            for w_shape in [[4096]]:
                for w_type in ['float32']:
                    for epilson in [1e-6]:
                        with paddle.pir_utils.IrGuard():
                            start_prog = paddle.static.Program()
                            main_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(
                                main_prog, start_prog
                            ):
                                residual = paddle.static.data(
                                    name='residual',
                                    shape=x_shape,
                                    dtype='float32',
                                )
                                x = paddle.static.data(
                                    name='x', shape=x_shape, dtype='float32'
                                )
                                w_attr = paddle.ParamAttr(
                                    learning_rate=0.0,
                                    initializer=paddle.nn.initializer.Normal(
                                        mean=0.0, std=2.0
                                    ),
                                )
                                add_out = paddle.add(residual, x)
                                layer_norm = paddle.nn.LayerNorm(
                                    add_out.shape[-1:],
                                    epsilon=epilson,
                                    weight_attr=w_attr,
                                )
                                out = layer_norm(add_out)
                                out = paddle.assign(out)
                                self.pass_list = ['add_norm_fuse_pass']
                                self.feeds = {
                                    "x": np.random.random(x_shape).astype(
                                        "float32"
                                    ),
                                    "residual": np.random.random(
                                        x_shape
                                    ).astype("float32"),
                                }
                                self.fetch_list = [out]
                                self.valid_op_map = {
                                    "pd_op.add": 0,
                                    "pd_op.layer_norm": 0,
                                    "pd_op.fused_bias_residual_layernorm": 1,
                                }

                                yield [main_prog, start_prog], False


if __name__ == "__main__":
    unittest.main()
