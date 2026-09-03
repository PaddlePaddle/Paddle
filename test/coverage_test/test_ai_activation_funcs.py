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

# [AUTO-GENERATED] Unit test for paddle.nn.functional.activation (static graph branches)
# 自动生成的单测，覆盖 paddle.nn.functional.activation 模块中未覆盖的静态图分支代码
# Target: cover uncovered lines 91-110, 152-164, 221-233, 278-290, 364-376, 418-443, 496-507
#   in paddle/python/paddle/nn/functional/activation.py
# 目标：覆盖 activation.py 中 celu, elu, hardshrink, hardtanh, hardsigmoid, hardswish, leaky_relu
#   等激活函数的旧静态图分支（else 分支）和 celu 的 alpha=0 错误处理

"""
This test covers the following modules and code paths:
这个测试覆盖以下模块和代码路径：

1. F.celu() - alpha=0 ZeroDivisionError (line 92), static graph branch (lines 99-110)
   F.celu() - alpha=0 时的零除错误以及静态图分支

2. F.elu() - static graph branch (lines 153-164)
   F.elu() - 静态图分支

3. F.hardshrink() - static graph branch (lines 222-233)
   F.hardshrink() - 静态图分支

4. F.hardtanh() - static graph branch (lines 278-290)
   F.hardtanh() - 静态图分支

5. F.hardsigmoid() - static graph branch (lines 364-376)
   F.hardsigmoid() - 静态图分支

6. F.hardswish() - static graph branch (lines 418-443)
   F.hardswish() - 静态图分支

7. F.leaky_relu() - static graph branch (lines 496-507)
   F.leaky_relu() - 静态图分支
"""

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F


class TestCeluAlphaZeroError(unittest.TestCase):
    """Test celu() raises ZeroDivisionError when alpha=0.
    测试 celu() 在 alpha=0 时抛出 ZeroDivisionError。
    覆盖 paddle/python/paddle/nn/functional/activation.py 第 91-92 行。
    """

    def test_celu_alpha_zero(self):
        """celu() should raise ZeroDivisionError when alpha is 0.
        当 alpha 为 0 时，celu() 应抛出 ZeroDivisionError。
        """
        x = paddle.to_tensor([-1.0, 0.0, 1.0])
        with self.assertRaises(ZeroDivisionError):
            F.celu(x, alpha=0)


class TestActivationStaticGraph(unittest.TestCase):
    """Test activation functions in static graph mode to cover the else branches.
    在静态图模式下测试激活函数以覆盖 else 分支。
    覆盖 paddle/python/paddle/nn/functional/activation.py 中多个激活函数的静态图路径。
    """

    def _run_static(self, func, input_data, **kwargs):
        """Helper to run a function in static graph mode.
        辅助方法：在静态图模式下运行函数。
        """
        paddle.enable_static()
        try:
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(
                    name='x', shape=input_data.shape, dtype='float32'
                )
                out = func(x, **kwargs)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_prog)
            result = exe.run(
                main_prog,
                feed={'x': input_data},
                fetch_list=[out],
            )
            return result[0]
        finally:
            paddle.disable_static()

    def test_celu_static(self):
        """Test celu in static graph mode.
        在静态图模式下测试 celu 激活函数。
        覆盖第 99-110 行。
        """
        input_data = np.array([[-0.2, 6.0], [1.0, 15.6]], dtype='float32')
        result = self._run_static(F.celu, input_data, alpha=1.0)
        expected = np.maximum(0, input_data) + np.minimum(
            0, 1.0 * (np.exp(input_data / 1.0) - 1)
        )
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_elu_static(self):
        """Test elu in static graph mode.
        在静态图模式下测试 elu 激活函数。
        覆盖第 153-164 行。
        """
        input_data = np.array([[-1.0, 6.0], [1.0, 15.6]], dtype='float32')
        result = self._run_static(F.elu, input_data, alpha=0.2)
        expected = np.where(
            input_data > 0, input_data, 0.2 * (np.exp(input_data) - 1)
        )
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_hardshrink_static(self):
        """Test hardshrink in static graph mode.
        在静态图模式下测试 hardshrink 激活函数。
        覆盖第 222-233 行。
        """
        input_data = np.array([-1.0, 0.3, 2.5], dtype='float32')
        result = self._run_static(F.hardshrink, input_data, threshold=0.5)
        expected = np.where(np.abs(input_data) > 0.5, input_data, 0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_hardtanh_static(self):
        """Test hardtanh in static graph mode.
        在静态图模式下测试 hardtanh 激活函数。
        覆盖第 278-290 行。
        """
        input_data = np.array([-1.5, 0.3, 2.5], dtype='float32')
        result = self._run_static(F.hardtanh, input_data, min=-1.0, max=1.0)
        expected = np.clip(input_data, -1.0, 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_hardsigmoid_static(self):
        """Test hardsigmoid in static graph mode.
        在静态图模式下测试 hardsigmoid 激活函数。
        覆盖第 364-376 行。
        """
        input_data = np.array([-4.0, 5.0, 1.0], dtype='float32')
        result = self._run_static(F.hardsigmoid, input_data)
        expected = np.clip(input_data / 6.0 + 0.5, 0, 1)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_hardswish_static(self):
        """Test hardswish in static graph mode.
        在静态图模式下测试 hardswish 激活函数。
        覆盖第 418-443 行。
        """
        input_data = np.array([-4.0, 5.0, 1.0], dtype='float32')
        result = self._run_static(F.hardswish, input_data)
        expected = np.where(
            input_data <= -3.0,
            0,
            np.where(
                input_data >= 3.0, input_data, input_data * (input_data + 3) / 6
            ),
        )
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_leaky_relu_static(self):
        """Test leaky_relu in static graph mode.
        在静态图模式下测试 leaky_relu 激活函数。
        覆盖第 496-507 行。
        """
        input_data = np.array([-1.0, 0.0, 1.0, -0.5, 2.0], dtype='float32')
        result = self._run_static(F.leaky_relu, input_data, negative_slope=0.01)
        expected = np.where(input_data >= 0, input_data, 0.01 * input_data)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_leaky_relu_static_custom_slope(self):
        """Test leaky_relu in static graph mode with custom slope.
        在静态图模式下使用自定义斜率测试 leaky_relu。
        """
        input_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype='float32')
        result = self._run_static(F.leaky_relu, input_data, negative_slope=0.1)
        expected = np.where(input_data >= 0, input_data, 0.1 * input_data)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
