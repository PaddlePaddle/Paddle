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

import sys
import unittest

import numpy as np

import paddle
import paddle.nn.functional as F


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or sys.platform == 'win32',
    "Skipping tests: CUDA is not available or running on Windows.",
)
class TestSlowConv2d(unittest.TestCase):
    def setUp(self):
        # Save old flag states
        self.old_flag_acc = paddle.get_flags(
            ['FLAGS_use_accuracy_compatible_kernel']
        )
        self.old_flag_disable = paddle.get_flags(['FLAGS_conv2d_disable_cudnn'])

        self.place = paddle.CUDAPlace(0)

        np.random.seed(2026)
        paddle.seed(2026)

    def tearDown(self):
        # Restore flags
        paddle.set_flags(self.old_flag_acc)
        paddle.set_flags(self.old_flag_disable)

    def _get_atol_rtol(self, dtype):
        if dtype == 'float64':
            return 1e-7, 1e-7
        elif dtype == 'float32':
            return 1e-4, 1e-4
        elif dtype == 'float16':
            return 5e-2, 5e-2
        return 1e-4, 1e-4

    def _init_data(self, dtype, layout, with_bias):
        groups = 1
        N = 2
        C_in = 4
        C_out = 4
        H, W = 8, 8
        K = 3

        if layout == "NCHW":
            input_shape = [N, C_in, H, W]
        else:  # NHWC
            input_shape = [N, H, W, C_in]

        weight_shape = [C_out, C_in // groups, K, K]

        np_x = np.random.randn(*input_shape).astype('float32')
        np_w = np.random.randn(*weight_shape).astype('float32')

        np_b = None
        if with_bias:
            np_b = np.random.randn(C_out).astype('float32')

        return np_x, np_w, np_b, groups

    def _run_op(
        self, np_x, np_w, np_b, dtype, layout, groups, disable_cudnn_flag
    ):
        # Control whether to use cuDNN via FLAGS_conv2d_disable_cudnn
        paddle.set_flags({'FLAGS_conv2d_disable_cudnn': disable_cudnn_flag})

        # Also ensure accuracy compatible kernel is enabled if needed
        paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': 1})

        x = paddle.to_tensor(np_x, place=self.place, dtype=dtype)
        x.stop_gradient = False

        w = paddle.to_tensor(np_w, place=self.place, dtype=dtype)
        w.stop_gradient = False

        b = None
        if np_b is not None:
            b = paddle.to_tensor(np_b, place=self.place, dtype=dtype)
            b.stop_gradient = False

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

        loss = out.sum()
        loss.backward()

        return {
            "out": out.cast('float32').numpy(),
            "x_grad": x.grad.cast('float32').numpy()
            if x.grad is not None
            else np.zeros_like(np_x),
            "w_grad": w.grad.cast('float32').numpy()
            if w.grad is not None
            else np.zeros_like(np_w),
            "b_grad": b.grad.cast('float32').numpy()
            if b is not None and b.grad is not None
            else None,
        }

    def _check_implementation(self, dtype, layout="NCHW", with_bias=True):
        np_x, np_w, np_b, groups = self._init_data(dtype, layout, with_bias)
        atol, rtol = self._get_atol_rtol(dtype)
        # -------------------------------------------------
        # Reference Run (Flag=0) ->  cuDNN Enabled
        # -------------------------------------------------
        res_ref = self._run_op(
            np_x, np_w, np_b, dtype, layout, groups, disable_cudnn_flag=0
        )

        # -------------------------------------------------
        # Target Run (Flag=1) ->  cuDNN Disabled (Fallback Kernel)
        # -------------------------------------------------
        res_tgt = self._run_op(
            np_x, np_w, np_b, dtype, layout, groups, disable_cudnn_flag=1
        )

        # -------------------------------------------------
        # Assertions
        # -------------------------------------------------
        msg = f"Failed at {dtype} with {layout}"

        np.testing.assert_allclose(
            res_tgt["out"],
            res_ref["out"],
            atol=atol,
            rtol=rtol,
            err_msg=f"{msg} (Forward)",
        )

        np.testing.assert_allclose(
            res_tgt["x_grad"],
            res_ref["x_grad"],
            atol=atol,
            rtol=rtol,
            err_msg=f"{msg} (X Grad)",
        )

        np.testing.assert_allclose(
            res_tgt["w_grad"],
            res_ref["w_grad"],
            atol=atol,
            rtol=rtol,
            err_msg=f"{msg} (W Grad)",
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
    # Test Cases for Registered Types
    # =================================================================
    def test_fp16(self):
        self._check_implementation('float16', layout="NCHW", with_bias=True)
        self._check_implementation('float16', layout="NCHW", with_bias=False)
        self._check_implementation('float16', layout="NHWC", with_bias=True)
        self._check_implementation('float16', layout="NHWC", with_bias=False)

    def test_fp32(self):
        self._check_implementation('float32', layout="NCHW", with_bias=True)
        self._check_implementation('float32', layout="NCHW", with_bias=False)
        self._check_implementation('float32', layout="NHWC", with_bias=True)
        self._check_implementation('float32', layout="NHWC", with_bias=False)

    def test_fp64(self):
        self._check_implementation('float64', layout="NCHW", with_bias=True)
        self._check_implementation('float64', layout="NCHW", with_bias=False)
        self._check_implementation('float64', layout="NHWC", with_bias=True)
        self._check_implementation('float64', layout="NHWC", with_bias=False)


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or sys.platform == 'win32',
    "Skipping tests: CUDA is not available or running on Windows.",
)
class TestSlowConv2dDilated(unittest.TestCase):
    def setUp(self):
        # Save old flag states
        self.old_flag_acc = paddle.get_flags(
            ['FLAGS_use_accuracy_compatible_kernel']
        )
        self.old_flag_disable = paddle.get_flags(['FLAGS_conv2d_disable_cudnn'])

        self.place = paddle.CUDAPlace(0)

        np.random.seed(2026)
        paddle.seed(2026)

    def tearDown(self):
        # Restore flags
        paddle.set_flags(self.old_flag_acc)
        paddle.set_flags(self.old_flag_disable)

    def _get_atol_rtol(self, dtype):
        if dtype == 'float64':
            return 1e-7, 1e-7
        elif dtype == 'float32':
            return 1e-4, 1e-4
        elif dtype == 'float16':
            # FP16 累加误差在 Native 和 cuDNN 之间可能较大，适当放宽
            return 5e-2, 5e-2
        return 1e-4, 1e-4

    def _init_data(self, dtype, layout, with_bias):
        groups = 1
        N = 2
        C_in = 4
        C_out = 4
        H, W = 16, 16  # 稍微加大一点尺寸以容纳 dilation
        K = 3

        if layout == "NCHW":
            input_shape = [N, C_in, H, W]
        else:  # NHWC
            input_shape = [N, H, W, C_in]

        weight_shape = [C_out, C_in // groups, K, K]

        np_x = np.random.randn(*input_shape).astype('float32')
        np_w = np.random.randn(*weight_shape).astype('float32')

        np_b = None
        if with_bias:
            np_b = np.random.randn(C_out).astype('float32')

        return np_x, np_w, np_b, groups

    def _run_op(
        self, np_x, np_w, np_b, dtype, layout, groups, disable_cudnn_flag
    ):
        # Control whether to use cuDNN via FLAGS_conv2d_disable_cudnn
        # Flag=0 -> Use cuDNN (Reference)
        # Flag=1 -> Disable cuDNN (Target, hits SlowDilated2d if implementation is correct)
        paddle.set_flags({'FLAGS_conv2d_disable_cudnn': disable_cudnn_flag})

        # Ensure accuracy compatible kernel is enabled to reduce non-deterministic algo noise
        paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': 1})

        x = paddle.to_tensor(np_x, place=self.place, dtype=dtype)
        x.stop_gradient = False

        w = paddle.to_tensor(np_w, place=self.place, dtype=dtype)
        w.stop_gradient = False

        b = None
        if np_b is not None:
            b = paddle.to_tensor(np_b, place=self.place, dtype=dtype)
            b.stop_gradient = False

        # [Key Configuration] Dilation = 2
        out = F.conv2d(
            x,
            w,
            b,
            stride=1,
            padding=1,
            dilation=2,  # <--- 强制走 dilated 逻辑
            groups=groups,
            data_format=layout,
        )

        loss = out.sum()
        loss.backward()

        return {
            "out": out.cast('float32').numpy(),
            "x_grad": x.grad.cast('float32').numpy()
            if x.grad is not None
            else np.zeros_like(np_x),
            "w_grad": w.grad.cast('float32').numpy()
            if w.grad is not None
            else np.zeros_like(np_w),
            "b_grad": b.grad.cast('float32').numpy()
            if b is not None and b.grad is not None
            else None,
        }

    def _check_implementation(self, dtype, layout="NCHW", with_bias=True):
        np_x, np_w, np_b, groups = self._init_data(dtype, layout, with_bias)
        atol, rtol = self._get_atol_rtol(dtype)

        # -------------------------------------------------
        # Reference Run (Flag=0) ->  cuDNN Enabled
        # -------------------------------------------------
        res_ref = self._run_op(
            np_x, np_w, np_b, dtype, layout, groups, disable_cudnn_flag=0
        )

        # -------------------------------------------------
        # Target Run (Flag=1) ->  cuDNN Disabled (Fallback Kernel)
        # -------------------------------------------------
        res_tgt = self._run_op(
            np_x, np_w, np_b, dtype, layout, groups, disable_cudnn_flag=1
        )

        # -------------------------------------------------
        # Assertions
        # -------------------------------------------------
        msg = f"Failed at {dtype} with {layout}"

        np.testing.assert_allclose(
            res_tgt["out"],
            res_ref["out"],
            atol=atol,
            rtol=rtol,
            err_msg=f"{msg} (Forward)",
        )

        np.testing.assert_allclose(
            res_tgt["x_grad"],
            res_ref["x_grad"],
            atol=atol,
            rtol=rtol,
            err_msg=f"{msg} (X Grad)",
        )

        np.testing.assert_allclose(
            res_tgt["w_grad"],
            res_ref["w_grad"],
            atol=atol,
            rtol=rtol,
            err_msg=f"{msg} (W Grad)",
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
    # Test Cases for Registered Types
    # =================================================================
    def test_fp16(self):
        self._check_implementation('float16', layout="NCHW", with_bias=True)
        self._check_implementation('float16', layout="NCHW", with_bias=False)
        self._check_implementation('float16', layout="NHWC", with_bias=True)
        self._check_implementation('float16', layout="NHWC", with_bias=False)

    def test_fp32(self):
        self._check_implementation('float32', layout="NCHW", with_bias=True)
        self._check_implementation('float32', layout="NCHW", with_bias=False)
        self._check_implementation('float32', layout="NHWC", with_bias=True)
        self._check_implementation('float32', layout="NHWC", with_bias=False)

    def test_fp64(self):
        self._check_implementation('float64', layout="NCHW", with_bias=True)
        self._check_implementation('float64', layout="NCHW", with_bias=False)
        self._check_implementation('float64', layout="NHWC", with_bias=True)
        self._check_implementation('float64', layout="NHWC", with_bias=False)


if __name__ == '__main__':
    unittest.main()
