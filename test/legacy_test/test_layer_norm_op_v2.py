# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import Program, core, program_guard


class TestDygraphLayerNormv2(unittest.TestCase):
    def test_dygraph(self):
        places = []
        if os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower() in [
            '1',
            'true',
            'on',
        ] or not (
            core.is_compiled_with_cuda() and core.op_support_gpu("layer_norm")
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda() and core.op_support_gpu("layer_norm"):
            places.append(base.CUDAPlace(0))
        for p in places:
            shape = [4, 10, 4, 4]

            def compute_v1(x):
                with base.dygraph.guard(p):
                    ln = paddle.nn.LayerNorm(shape[1:])
                    y = ln(paddle.to_tensor(x))
                return y.numpy()

            def compute_v2(x):
                with base.dygraph.guard(p):
                    ln = paddle.nn.LayerNorm(shape[1:])
                    y = ln(paddle.to_tensor(x))
                return y.numpy()

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)

    def test_eager(self):
        places = []
        if os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower() in [
            '1',
            'true',
            'on',
        ] or not (
            core.is_compiled_with_cuda() and core.op_support_gpu("layer_norm")
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda() and core.op_support_gpu("layer_norm"):
            places.append(base.CUDAPlace(0))
        for p in places:
            shape = [4, 10, 4, 4]

            def compute_v1(x):
                with base.dygraph.guard(p):
                    ln = paddle.nn.LayerNorm(shape[1:])
                    x1 = paddle.to_tensor(x)
                    x1.stop_gradient = False
                    y = ln(x1)
                    y.backward()
                    return y.numpy(), x1.gradient()

            def compute_v2(x):
                with base.dygraph.guard(p):
                    ln = paddle.nn.LayerNorm(shape[1:])
                    x1 = paddle.to_tensor(x)
                    x1.stop_gradient = False
                    y = ln(x1)
                    y.backward()
                    return y.numpy(), x1.gradient()

            x = np.random.randn(*shape).astype("float32")
            y1, g1 = compute_v1(x)
            y2, g2 = compute_v2(x)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)
            np.testing.assert_allclose(g1, g2, rtol=1e-05)

    def test_static(self):
        paddle.enable_static()
        places = []
        if os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower() in [
            '1',
            'true',
            'on',
        ] or not (
            core.is_compiled_with_cuda() and core.op_support_gpu("layer_norm")
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda() and core.op_support_gpu("layer_norm"):
            places.append(base.CUDAPlace(0))
        for p in places:
            exe = base.Executor(p)
            shape = [4, 10, 16, 16]

            def compute_v1(x_np):
                with program_guard(Program(), Program()):
                    ln = paddle.nn.LayerNorm(shape[1:])
                    x = paddle.static.data(
                        name='x', shape=x_np.shape, dtype=x_np.dtype
                    )
                    y = ln(x)
                    exe.run(base.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            def compute_v2(x_np):
                with program_guard(Program(), Program()):
                    ln = paddle.nn.LayerNorm(shape[1:])
                    x = paddle.static.data(
                        name='x', shape=x_np.shape, dtype=x_np.dtype
                    )
                    y = ln(x)
                    exe.run(base.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)


class TestLayerNormFunction(unittest.TestCase):
    def test_dygraph(self):
        places = []
        if os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower() in [
            '1',
            'true',
            'on',
        ] or not (
            core.is_compiled_with_cuda() and core.op_support_gpu("layer_norm")
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda() and core.op_support_gpu("layer_norm"):
            places.append(base.CUDAPlace(0))
        for p in places:
            shape = [4, 10, 4, 4]

            def compute_v0(x):
                with base.dygraph.guard(p):
                    ln = paddle.nn.LayerNorm(shape[1:])
                    y = ln(paddle.to_tensor(x))
                return y.numpy()

            def compute_v1(x):
                with base.dygraph.guard(p):
                    x = paddle.to_tensor(x)
                    y = paddle.nn.functional.layer_norm(x, shape[1:])
                return y.numpy()

            def compute_v2(x):
                with base.dygraph.guard(p):
                    x = paddle.to_tensor(x)
                    y = paddle.nn.functional.layer_norm(x, tuple(shape[1:]))
                return y.numpy()

            def compute_v3(x):
                with base.dygraph.guard(p):
                    ln = paddle.nn.LayerNorm(shape[-1])
                    y = ln(paddle.to_tensor(x))
                return y.numpy()

            def compute_v4(x):
                with base.dygraph.guard(p):
                    x = paddle.to_tensor(x)
                    y = paddle.nn.functional.layer_norm(x, shape[-1])
                return y.numpy()

            x = np.random.randn(*shape).astype("float32")
            y0 = compute_v0(x)
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            np.testing.assert_allclose(y0, y1, rtol=1e-05)
            np.testing.assert_allclose(y0, y2, rtol=1e-05)
            y3 = compute_v3(x)
            y4 = compute_v4(x)
            np.testing.assert_allclose(y3, y4, rtol=1e-05)

            self.assertRaises(
                ValueError,
                paddle.nn.functional.layer_norm,
                x=x,
                normalized_shape=1.0,
            )


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
