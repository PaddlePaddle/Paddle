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
                                self.pass_attr_list = [
                                    {'add_norm_fuse_pass': {}}
                                ]
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
                                self.pass_attr_list = [
                                    {'add_norm_fuse_pass': {}}
                                ]
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


class TestAddRmsNormFusePatternWithResidual(TestRmsNormFusePattern):
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
                                w1 = create_parameter(
                                    name="w1",
                                    shape=w_shape,
                                    dtype=w_type,
                                    initializer=paddle.nn.initializer.Assign(
                                        np.random.random([4096, 4096]).astype(
                                            w_type
                                        )
                                    ),
                                )
                                add_out = paddle.add(residual, x)
                                add_out_1 = add_out
                                variance = add_out.pow(2).mean(-1, keepdim=True)
                                add_out = (
                                    paddle.rsqrt(variance + 1e-6) * add_out
                                )
                                mul_out = add_out * w
                                matmul_out = paddle.matmul(mul_out, w1)
                                out = paddle.add(add_out_1, matmul_out)
                                out = paddle.assign(out)
                                self.pass_attr_list = [
                                    {'add_norm_fuse_pass': {}}
                                ]
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
                for x_type in ['float32', 'float16']:
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
                                    dtype=x_type,
                                )
                                x = paddle.static.data(
                                    name='x', shape=x_shape, dtype=x_type
                                )
                                w_attr = paddle.ParamAttr(
                                    learning_rate=0.0,
                                    initializer=paddle.nn.initializer.Normal(
                                        mean=0.0, std=2.0
                                    ),
                                )
                                b_attr = paddle.ParamAttr(
                                    learning_rate=0.0,
                                    initializer=paddle.nn.initializer.Normal(
                                        mean=0.0, std=2.0
                                    ),
                                )
                                w1 = create_parameter(
                                    name="w1",
                                    shape=w_shape,
                                    dtype=x_type,
                                    initializer=paddle.nn.initializer.Assign(
                                        np.random.random([4096, 4096]).astype(
                                            x_type
                                        )
                                    ),
                                )
                                add_out = paddle.add(residual, x)
                                add_out_1 = add_out
                                layer_norm = paddle.nn.LayerNorm(
                                    add_out.shape[-1:],
                                    epsilon=epilson,
                                    weight_attr=w_attr,
                                    bias_attr=b_attr,
                                )
                                layer_norm_out = layer_norm(add_out)
                                matmul_out = paddle.matmul(layer_norm_out, w1)
                                out = paddle.add(add_out_1, matmul_out)
                                out = paddle.assign(out)
                                self.pass_attr_list = [
                                    {'add_norm_fuse_pass': {}}
                                ]
                                self.feeds = {
                                    "x": np.random.random(x_shape).astype(
                                        x_type
                                    ),
                                    "residual": np.random.random(
                                        x_shape
                                    ).astype(x_type),
                                }
                                self.fetch_list = [out]
                                self.valid_op_map = {
                                    "pd_op.layer_norm": 0,
                                    "pd_op.fused_bias_residual_layernorm": 1,
                                }

                                yield [main_prog, start_prog], False

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)


class TestAddGroupNormPattern_FP16(PassTest):
    r"""
    x         residual
    |           |
         add
          |
      group_norm
    """

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for x_shape in [[2, 6, 4, 2]]:
            for residual_shape in [[1, 6, 1, 1]]:
                for dtype in ['float16']:
                    for epilson in [1e-5]:
                        for groups in [2]:
                            for data_layout in ['NCHW']:
                                rand_value = (
                                    0.001
                                    * paddle.rand(
                                        shape=[x_shape[1]], dtype=dtype
                                    ).numpy()
                                )
                                with paddle.pir_utils.IrGuard():
                                    start_prog = paddle.static.Program()
                                    main_prog = paddle.static.Program()
                                    with paddle.pir.core.program_guard(
                                        main_prog, start_prog
                                    ):
                                        residual = paddle.static.data(
                                            name='residual',
                                            shape=residual_shape,
                                            dtype=dtype,
                                        )
                                        x = paddle.static.data(
                                            name='x', shape=x_shape, dtype=dtype
                                        )
                                        w = create_parameter(
                                            shape=[x_shape[1]],
                                            dtype=dtype,
                                            initializer=paddle.nn.initializer.Assign(
                                                rand_value
                                            ),
                                        )
                                        b = create_parameter(
                                            shape=[residual_shape[1]],
                                            dtype=dtype,
                                            initializer=paddle.nn.initializer.Assign(
                                                rand_value
                                            ),
                                        )
                                        add_out = paddle.add(x, residual)

                                        group_norm_out = (
                                            paddle.nn.functional.group_norm(
                                                add_out,
                                                num_groups=groups,
                                                epsilon=epilson,
                                                weight=w,
                                                bias=b,
                                                data_format=data_layout,
                                            )
                                        )
                                        out = paddle.assign(group_norm_out)
                                        self.pass_attr_list = [
                                            {'add_norm_fuse_pass': {}},
                                            {'transfer_layout_pass': {}},
                                            {
                                                'remove_redundant_transpose_pass': {}
                                            },
                                        ]
                                        self.feeds = {
                                            "x": np.random.random(
                                                x_shape
                                            ).astype(dtype),
                                            "residual": np.random.random(
                                                residual_shape
                                            ).astype(dtype),
                                        }
                                        self.fetch_list = [out]
                                        self.valid_op_map = {
                                            "pa_op.add": 0,
                                            "pd_op.group_norm": 0,
                                            "pd_op.add_group_norm_silu": 1,
                                        }
                                        yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


class TestAddGroupNormPatternSilu_FP16(PassTest):
    r"""
    x         residual
    |           |
         add
          |
      group_norm
    """

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for x_shape in [[2, 6, 4, 2]]:
            for residual_shape in [[1, 6, 1, 1]]:
                for dtype in ['float16']:
                    for epilson in [1e-5]:
                        for groups in [2]:
                            for data_layout in ['NCHW']:
                                rand_value = (
                                    0.001
                                    * paddle.rand(
                                        shape=[x_shape[1]], dtype=dtype
                                    ).numpy()
                                )
                                with paddle.pir_utils.IrGuard():
                                    start_prog = paddle.static.Program()
                                    main_prog = paddle.static.Program()
                                    with paddle.pir.core.program_guard(
                                        main_prog, start_prog
                                    ):
                                        residual = paddle.static.data(
                                            name='residual',
                                            shape=residual_shape,
                                            dtype=dtype,
                                        )
                                        x = paddle.static.data(
                                            name='x', shape=x_shape, dtype=dtype
                                        )
                                        w = create_parameter(
                                            shape=[x_shape[1]],
                                            dtype=dtype,
                                            initializer=paddle.nn.initializer.Assign(
                                                rand_value
                                            ),
                                        )
                                        b = create_parameter(
                                            shape=[x_shape[1]],
                                            dtype=dtype,
                                            initializer=paddle.nn.initializer.Assign(
                                                rand_value
                                            ),
                                        )
                                        add_out = paddle.add(x, residual)
                                        group_norm_out = (
                                            paddle.nn.functional.group_norm(
                                                add_out,
                                                num_groups=groups,
                                                epsilon=epilson,
                                                weight=w,
                                                bias=b,
                                                data_format=data_layout,
                                            )
                                        )
                                        out = paddle.nn.functional.silu(
                                            group_norm_out
                                        )
                                        out = paddle.assign(out)
                                        self.pass_attr_list = [
                                            {'add_norm_fuse_pass': {}},
                                            {'transfer_layout_pass': {}},
                                            {
                                                'remove_redundant_transpose_pass': {}
                                            },
                                        ]
                                        self.feeds = {
                                            "x": np.random.random(
                                                x_shape
                                            ).astype(dtype),
                                            "residual": np.random.random(
                                                residual_shape
                                            ).astype(dtype),
                                        }
                                        self.fetch_list = [out]
                                        self.valid_op_map = {
                                            "pd_op.silu": 0,
                                            "pd_op.add": 0,
                                            "pd_op.group_norm": 0,
                                            "pd_op.add_group_norm_silu": 1,
                                        }
                                        yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
