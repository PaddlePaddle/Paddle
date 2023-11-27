#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
from scipy import special

import paddle
from paddle.base import core
from paddle.pir_utils import test_with_pir_api

np.random.seed(42)
paddle.seed(42)


def reference_i1e(x):
    return special.i1e(x)


def reference_i1e_grad(x, dout):
    eps = np.finfo(x.dtype).eps
    not_tiny = abs(x) > eps
    safe_x = np.where(not_tiny, x, eps)
    gradx = special.i0e(safe_x) - special.i1e(x) * (np.sign(x) + 1 / safe_x)
    gradx = np.where(not_tiny, gradx, 0.5)
    return dout * gradx


class TestI1e_API(unittest.TestCase):
    DTYPE = "float64"
    DATA = [0, 1, 2, 3, 4, 5]

    def setUp(self):
        self.x = np.array(self.DATA).astype(self.DTYPE)
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_api_static(self):
        @test_with_pir_api
        def run(place):
            paddle.enable_static()
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data(
                    name="x", shape=self.x.shape, dtype=self.DTYPE
                )
                y = paddle.i1e(x)
                exe = paddle.static.Executor(place)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x": self.x},
                    fetch_list=[y],
                )
                out_ref = reference_i1e(self.x)
                np.testing.assert_allclose(out_ref, res[0], rtol=1e-5)
            paddle.disable_static()

        for place in self.place:
            run(place)

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x)
            out = paddle.i1e(x)

            out_ref = reference_i1e(self.x)
            np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-5)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_empty_input_error(self):
        for place in self.place:
            paddle.disable_static(place)
            x = None
            self.assertRaises(ValueError, paddle.i1e, x)
            paddle.enable_static()


class Testi1eFloat32Zero2EightCase(TestI1e_API):
    DTYPE = "float32"
    DATA = [0, 1, 2, 3, 4, 5, 6, 7, 8]


class Testi1eFloat32OverEightCase(TestI1e_API):
    DTYPE = "float32"
    DATA = [9, 10, 11, 12, 13, 14, 15, 16, 17]


class Testi1eFloat64Zero2EightCase(TestI1e_API):
    DTYPE = "float64"
    DATA = [0, 1, 2, 3, 4, 5, 6, 7, 8]


class Testi1eFloat64OverEightCase(TestI1e_API):
    DTYPE = "float64"
    DATA = [9, 10, 11, 12, 13, 14, 15, 16, 17]


class TestI1eOp(OpTest):
    # 配置 op 信息以及输入输出等参数
    def setUp(self):
        self.op_type = "i1e"
        self.python_api = paddle.i1e
        self.init_config()
        self.outputs = {'out': self.target}

    # 测试前向输出结果
    def test_check_output(self):
        self.check_output(check_pir=True)

    # 测试反向梯度输出
    def test_check_grad(self):
        self.check_grad(
            ['x'],
            'out',
            user_defined_grads=[
                reference_i1e_grad(
                    self.case,
                    1 / self.case.size,
                )
            ],
            check_pir=True,
        )

    # 生成随机的输入数据并计算对应输出
    def init_config(self):
        zero_case = np.zeros(1).astype('float64')
        rand_case = np.random.randn(250).astype('float64')
        over_eight_case = np.random.uniform(low=8, high=9, size=250).astype(
            'float64'
        )
        self.case = np.concatenate([zero_case, rand_case, over_eight_case])
        self.inputs = {'x': self.case}
        self.target = reference_i1e(self.inputs['x'])


if __name__ == "__main__":
    unittest.main()
