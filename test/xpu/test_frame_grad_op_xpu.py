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
from op_test import convert_float_to_uint16, convert_uint16_to_float

import paddle
from paddle.base import core


def frame_grad_numpy(out_grad, x_shape, frame_length, hop_length, axis):
    x_grad = np.zeros(x_shape, dtype=out_grad.dtype)
    if axis == -1:
        n_frames = out_grad.shape[-1]
        for n in range(n_frames):
            start = n * hop_length
            end = start + frame_length
            x_grad[..., start:end] += out_grad[..., :, n]
    else:
        n_frames = out_grad.shape[0]
        for n in range(n_frames):
            start = n * hop_length
            end = start + frame_length
            x_grad[start:end, ...] += out_grad[n, ...]
    return x_grad


class TestFrameGradAPIXPU(unittest.TestCase):
    def setUp(self):
        if not core.is_compiled_with_xpu():
            self.skipTest("XPU is not available")
        paddle.disable_static()
        paddle.set_device("xpu")
        self.place = paddle.XPUPlace(0)
        np.random.seed(2026)

    def tearDown(self):
        paddle.enable_static()

    def _dtype_list(self):
        dtypes = ["float32", "float64", "float16"]
        if core.is_bfloat16_supported(self.place):
            dtypes.append("bfloat16")
        return dtypes

    def _make_tensor(self, x_np, dtype, stop_gradient):
        # bf16/float16 统一从 float32 源构造，避免 numpy 端缺 dtype 或精度噪声过大。
        if dtype in ["float16", "bfloat16"]:
            t = paddle.to_tensor(x_np.astype("float32"))
            t = paddle.cast(t, dtype)
        else:
            t = paddle.to_tensor(x_np.astype(dtype))
        t.stop_gradient = stop_gradient
        return t

    def _to_float_np(self, t):
        if t.dtype in [paddle.float16, paddle.bfloat16]:
            return paddle.cast(t, "float32").numpy()
        return t.numpy()

    def _bf16_quantize_np(self, x):
        x_fp32 = x.astype("float32", copy=False)
        return convert_uint16_to_float(convert_float_to_uint16(x_fp32))

    def run_case(self, shape, frame_length, hop_length, axis, dtype, seed):
        rng = np.random.default_rng(seed)

        x_np = rng.uniform(-1.0, 1.0, size=shape)
        if dtype == "bfloat16":
            x_np = self._bf16_quantize_np(x_np)
        x = self._make_tensor(x_np, dtype, stop_gradient=False)
        out = paddle.signal.frame(x, frame_length, hop_length, axis)

        out_grad_np = rng.uniform(-1.0, 1.0, size=tuple(out.shape))
        if dtype == "bfloat16":
            out_grad_np = self._bf16_quantize_np(out_grad_np)
        out_grad = self._make_tensor(out_grad_np, dtype, stop_gradient=True)
        (x_grad,) = paddle.grad([out], [x], grad_outputs=[out_grad])

        # 参考实现统一在高精度侧计算，再按对比口径落到 float32/float64。
        if dtype == "float64":
            expect = frame_grad_numpy(
                out_grad_np.astype("float64"),
                tuple(np.array(shape).tolist()),
                frame_length,
                hop_length,
                axis,
            )
            got = x_grad.numpy()
            rtol, atol = 1e-10, 1e-10
        elif dtype == "float32":
            expect = frame_grad_numpy(
                out_grad_np.astype("float32"),
                tuple(np.array(shape).tolist()),
                frame_length,
                hop_length,
                axis,
            )
            got = x_grad.numpy()
            rtol, atol = 1e-5, 1e-5
        elif dtype == "float16":
            expect = frame_grad_numpy(
                out_grad_np.astype("float32"),
                tuple(np.array(shape).tolist()),
                frame_length,
                hop_length,
                axis,
            ).astype("float32")
            got = self._to_float_np(x_grad)
            rtol, atol = 8e-3, 8e-3
        else:  # bfloat16
            expect = frame_grad_numpy(
                out_grad_np.astype("float32"),
                tuple(np.array(shape).tolist()),
                frame_length,
                hop_length,
                axis,
            ).astype("float32")
            got = self._to_float_np(x_grad)
            rtol, atol = 1.5e-2, 1.5e-2

        np.testing.assert_allclose(got, expect, rtol=rtol, atol=atol)

    def test_frame_grad_all_dtypes_and_cases(self):
        # 覆盖 axis0/axis-1、overlap/no-overlap、frame_length=1、frame_length=seq、hop=1/2/3/5。
        cases = [
            ((4, 17), 4, 2, -1),
            ((17, 4), 4, 2, 0),
            ((50,), 2, 3, -1),
            ((1, 127), 1, 1, -1),
            ((2, 129), 2, 1, -1),
            ((129, 2), 2, 1, 0),
            ((2, 19), 4, 5, -1),
            ((19, 2), 4, 5, 0),
            ((2, 5, 33), 2, 3, -1),
            ((33, 2, 5), 2, 3, 0),
            ((2, 31), 31, 7, -1),
            ((31, 2), 31, 7, 0),
            ((3, 33), 2, 3, -1),
            ((33, 3), 2, 3, 0),
        ]

        seed = 3100
        for dtype in self._dtype_list():
            for shape, frame_length, hop_length, axis in cases:
                with self.subTest(
                    dtype=dtype,
                    shape=shape,
                    frame_length=frame_length,
                    hop_length=hop_length,
                    axis=axis,
                ):
                    self.run_case(
                        shape, frame_length, hop_length, axis, dtype, seed
                    )
                    seed += 1


if __name__ == "__main__":
    unittest.main()
