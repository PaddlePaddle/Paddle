# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


class TestDy2staticNewIR(unittest.TestCase):
    def test_basic_network(self):
        def func(x):
            out = paddle.mean(x)
            return out

        static_func = paddle.jit.to_static(func)
        x = paddle.randn((3, 3))
        y = paddle.randn((3, 3))
        x.stop_gradient = False
        y.stop_gradient = False
        ans = func(x)
        out = static_func(x)

        np.testing.assert_allclose(
            out.numpy(), ans.numpy(), rtol=1e-05, atol=1e-8
        )

    def test_basic_network_backward(self):
        def func(x):
            out = paddle.mean(x)
            return out

        # ==== dygraph computation ====
        static_func = paddle.jit.to_static(func)
        x = paddle.randn((3, 3))
        y = paddle.randn((3, 3))
        x.stop_gradient = False
        y.stop_gradient = False
        loss = func(x) * 2
        loss.backward()
        x_grad_ans = x.grad.numpy()
        x.clear_gradient()

        # ==== to static compuatation ====
        out = static_func(x)
        out = out * 2
        out.backward()
        st_grad = x.grad

        np.testing.assert_allclose(
            x_grad_ans, st_grad.numpy(), rtol=1e-05, atol=1e-8
        )


class TestDy2staticNewIR3(unittest.TestCase):
    def test_complex_layer(self):
        def output_pure_func(x, y):
            outx = paddle.mean(x)
            outy = paddle.mean(y)
            outy.stop_gradient = True
            return paddle.add(outx, outy), outy

        def run_function(to_static=True):
            import paddle

            # 设置随机种子
            paddle.seed(2023)
            # 生成随机数
            x = paddle.randn((10, 10))
            y = paddle.randn((10, 10))
            x.stop_gradient = False
            y.stop_gradient = True
            func = output_pure_func
            if to_static:
                func = paddle.jit.to_static(func)
            y, y_mean = func(x, y)
            loss = y.mean()
            loss.backward()
            return (y, x.grad)

        for dy, st in zip(run_function(False), run_function(True)):
            np.testing.assert_allclose(
                dy.numpy(), st.numpy(), rtol=1e-05, atol=1e-8
            )


if __name__ == "__main__":
    unittest.main()
