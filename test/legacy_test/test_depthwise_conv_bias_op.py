# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle.static import InputSpec


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or sys.platform == 'win32',
    "Skipping tests: CUDA is not available or running on Windows.",
)
class TestDepthwiseConvBiasUnified(unittest.TestCase):
    def setUp(self):
        self.old_flag = paddle.get_flags(
            ['FLAGS_use_accuracy_compatible_kernel']
        )
        paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': 1})
        self.place = paddle.CUDAPlace(0)

    def tearDown(self):
        paddle.set_flags(self.old_flag)

    def _get_atol_rtol(self, dtype):
        if dtype == 'float64':
            return 1e-7, 1e-7
        elif dtype == 'float32':
            return 1e-4, 1e-4
        elif dtype == 'float16':
            return 5e-2, 5e-2

        return 1e-5, 1e-5

    def _init_data(self, dim, dtype, layout, with_bias):
        groups = 4
        C = groups
        K = 3

        if dim == 2:
            N, H, W = 2, 32, 32
            if layout == "NCHW":
                input_shape = [N, C, H, W]
            else:
                input_shape = [N, H, W, C]
            weight_shape = [C, 1, K, K]

        elif dim == 3:
            N, D, H, W = 2, 8, 16, 16
            if layout == "NCDHW":
                input_shape = [N, C, D, H, W]
            else:
                input_shape = [N, D, H, W, C]
            weight_shape = [C, 1, K, K, K]
        else:
            raise ValueError(f"Unsupported dim: {dim}")

        elem_x = np.prod(input_shape)
        np_x = np.sin(np.arange(elem_x)).reshape(input_shape).astype('float32')

        elem_w = np.prod(weight_shape)
        np_w = np.cos(np.arange(elem_w)).reshape(weight_shape).astype('float32')
        np_b = None
        if with_bias:
            np_b = np.sin(np.arange(C)).astype('float32')

        return np_x, np_w, np_b, groups

    def _run_op(self, dim, np_x, np_w, np_b, dtype, layout, groups, flag_val):
        paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': flag_val})

        x = paddle.to_tensor(
            np_x, dtype=dtype, place=self.place, stop_gradient=False
        )
        w = paddle.to_tensor(
            np_w, dtype=dtype, place=self.place, stop_gradient=False
        )
        b = None
        if np_b is not None:
            b = paddle.to_tensor(
                np_b, dtype=dtype, place=self.place, stop_gradient=False
            )

        if dim == 2:
            out = F.conv2d(
                x,
                w,
                b,
                stride=1,
                padding=1,
                dilation=1,
                groups=groups,
                data_format=layout,
            )
        else:
            out = F.conv3d(
                x,
                w,
                b,
                stride=1,
                padding=1,
                dilation=1,
                groups=groups,
                data_format=layout,
            )

        loss = out.sum()
        loss.backward()

        return {
            "out": out.numpy(),
            "x_grad": x.grad.numpy(),
            "w_grad": w.grad.numpy(),
            "b_grad": b.grad.numpy() if b is not None else None,
        }

    def _check_forward(self, dim, dtype, layout, with_bias):
        np_x, np_w, np_b, groups = self._init_data(
            dim, dtype, layout, with_bias
        )
        atol, rtol = self._get_atol_rtol(dtype)

        res_ref = self._run_op(dim, np_x, np_w, np_b, dtype, layout, groups, 0)
        res_tgt = self._run_op(dim, np_x, np_w, np_b, dtype, layout, groups, 1)

        bias_str = "WithBias" if with_bias else "NoBias"
        msg = f"[Forward] {dim}D {bias_str}, dtype={dtype}, layout={layout}"

        np.testing.assert_allclose(
            res_tgt["out"], res_ref["out"], atol=atol, rtol=rtol, err_msg=msg
        )

    def _check_backward(self, dim, dtype, layout, with_bias):
        np_x, np_w, np_b, groups = self._init_data(
            dim, dtype, layout, with_bias
        )
        atol, rtol = self._get_atol_rtol(dtype)

        res_ref = self._run_op(dim, np_x, np_w, np_b, dtype, layout, groups, 0)
        res_tgt = self._run_op(dim, np_x, np_w, np_b, dtype, layout, groups, 1)

        bias_str = "WithBias" if with_bias else "NoBias"
        msg = f"[Backward] {dim}D {bias_str}, dtype={dtype}, layout={layout}"

        np.testing.assert_allclose(
            res_tgt["x_grad"],
            res_ref["x_grad"],
            atol=atol,
            rtol=rtol,
            err_msg=f"{msg} (Input Grad)",
        )
        np.testing.assert_allclose(
            res_tgt["w_grad"],
            res_ref["w_grad"],
            atol=atol,
            rtol=rtol,
            err_msg=f"{msg} (Weight Grad)",
        )
        if with_bias:
            np.testing.assert_allclose(
                res_tgt["b_grad"],
                res_ref["b_grad"],
                atol=atol,
                rtol=rtol,
                err_msg=f"{msg} (Bias Grad)",
            )

    # =================================================================
    # 2D Tests (FP32, FP64, FP16)
    # =================================================================
    def test_2d_fp32_forward(self):
        self._check_forward(2, 'float32', 'NCHW', True)
        self._check_forward(2, 'float32', 'NHWC', False)

    def test_2d_fp32_backward(self):
        self._check_backward(2, 'float32', 'NCHW', True)
        self._check_backward(2, 'float32', 'NHWC', False)

    def test_2d_fp64_forward(self):
        self._check_forward(2, 'float64', 'NCHW', True)
        self._check_forward(2, 'float64', 'NHWC', False)

    def test_2d_fp64_backward(self):
        self._check_backward(2, 'float64', 'NCHW', True)
        self._check_backward(2, 'float64', 'NHWC', False)

    def test_2d_fp16_forward(self):
        self._check_forward(2, 'float16', 'NCHW', True)
        self._check_forward(2, 'float16', 'NHWC', True)

    def test_2d_fp16_backward(self):
        self._check_backward(2, 'float16', 'NCHW', True)
        self._check_backward(2, 'float16', 'NHWC', True)

    # =================================================================
    # 3D Tests (FP32, FP64, FP16)
    # =================================================================
    def test_3d_fp32_forward(self):
        self._check_forward(3, 'float32', 'NCDHW', True)
        self._check_forward(3, 'float32', 'NDHWC', False)

    def test_3d_fp32_backward(self):
        self._check_backward(3, 'float32', 'NCDHW', True)
        self._check_backward(3, 'float32', 'NDHWC', False)

    def test_3d_fp64_forward(self):
        self._check_forward(3, 'float64', 'NCDHW', True)

    def test_3d_fp64_backward(self):
        self._check_backward(3, 'float64', 'NCDHW', True)

    def test_3d_fp16_forward(self):
        self._check_forward(3, 'float16', 'NCDHW', True)
        self._check_forward(3, 'float16', 'NDHWC', True)

    def test_3d_fp16_backward(self):
        self._check_backward(3, 'float16', 'NCDHW', True)
        self._check_backward(3, 'float16', 'NDHWC', True)


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or sys.platform == 'win32',
    "Skipping tests: CUDA is not available or running on Windows.",
)
class TestDepthwiseConvBiasSymbolicShape(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.old_flags = paddle.get_flags(
            ['FLAGS_use_accuracy_compatible_kernel']
        )
        paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': 1})

        self.env_key = 'MIN_GRAPH_SIZE'
        self.old_env_val = os.environ.get(self.env_key)
        os.environ[self.env_key] = '0'

        self.place = paddle.CUDAPlace(0)

    def tearDown(self):
        paddle.set_flags(self.old_flags)

        if self.old_env_val is not None:
            os.environ[self.env_key] = self.old_env_val
        else:
            if self.env_key in os.environ:
                del os.environ[self.env_key]

    def _run_symbolic_shape_check(self, dim, with_bias):
        groups = 4
        C = groups

        class ConvModel(paddle.nn.Layer):
            def __init__(self, dim, groups):
                super().__init__()
                self.dim = dim
                self.groups = groups

            def forward(self, x, w, b=None):
                if self.dim == 2:
                    out = F.conv2d(
                        x,
                        w,
                        b,
                        groups=self.groups,
                        padding=1,
                        data_format="NCHW",
                    )
                else:
                    out = F.conv3d(
                        x,
                        w,
                        b,
                        groups=self.groups,
                        padding=1,
                        data_format="NCDHW",
                    )
                return out

        if dim == 2:
            x_spec = InputSpec(
                shape=[None, C, None, None], dtype='float32', name='x'
            )
            w_spec = InputSpec(shape=[C, 1, 3, 3], dtype='float32', name='w')
        else:
            x_spec = InputSpec(
                shape=[None, C, None, None, None], dtype='float32', name='x'
            )
            w_spec = InputSpec(shape=[C, 1, 3, 3, 3], dtype='float32', name='w')

        b_spec = (
            InputSpec(shape=[C], dtype='float32', name='b')
            if with_bias
            else None
        )
        input_specs = [x_spec, w_spec]
        if with_bias:
            input_specs.append(b_spec)

        model = ConvModel(dim, groups)
        static_model = paddle.jit.to_static(
            model, input_spec=input_specs, backend="CINN", full_graph=True
        )

        batch_size = 2
        spatial_size = 16 if dim == 2 else 8
        elem_x = (
            batch_size * C * spatial_size * spatial_size
            if dim == 2
            else batch_size * C * spatial_size * spatial_size * spatial_size
        )
        x_shape = (
            (batch_size, C, spatial_size, spatial_size)
            if dim == 2
            else (batch_size, C, spatial_size, spatial_size, spatial_size)
        )

        np_x = np.sin(np.arange(elem_x)).reshape(x_shape).astype('float32')

        elem_w = np.prod(w_spec.shape)
        np_w = np.cos(np.arange(elem_w)).reshape(w_spec.shape).astype('float32')

        x_tensor = paddle.to_tensor(np_x, stop_gradient=False)
        w_tensor = paddle.to_tensor(np_w, stop_gradient=False)
        inputs = [x_tensor, w_tensor]
        if with_bias:
            b_tensor = paddle.to_tensor(
                np.random.randn(C).astype('float32'), stop_gradient=False
            )
            inputs.append(b_tensor)

        out = static_model(*inputs)

        loss = out.mean()
        loss.backward()

        self.assertIsNotNone(out)
        self.assertIsNotNone(x_tensor.grad)

    def test_depthwise_conv2d_bias_symbolic_forward_backward(self):
        self._run_symbolic_shape_check(dim=2, with_bias=True)

    def test_depthwise_conv3d_bias_symbolic_forward_backward(self):
        self._run_symbolic_shape_check(dim=3, with_bias=True)


if __name__ == '__main__':
    unittest.main()
