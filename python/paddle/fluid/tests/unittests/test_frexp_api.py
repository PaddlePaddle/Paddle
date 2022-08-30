# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import print_function
import unittest
import numpy as np
import paddle
import paddle.tensor.math as math
import paddle.fluid


def frexp(x):
    if x.dtype not in [paddle.float32, paddle.float64]:
        raise TypeError(
            "The data type of input must be one of ['float32', 'float64'], but got {}"
            .format(x.dtype))
    input_x = paddle.abs(x)
    exponent = paddle.floor(paddle.log2(input_x))
    exponent = paddle.where(paddle.isinf(exponent),
                            paddle.full_like(exponent, 0), exponent)

    # 0填充
    mantissa = paddle.divide(input_x, 2**exponent)
    # 计算exponent
    exponent = paddle.where((mantissa <= -1),
                            paddle.add(exponent, paddle.ones_like(exponent)),
                            exponent)
    exponent = paddle.where((mantissa >= 1),
                            paddle.add(exponent, paddle.ones_like(exponent)),
                            exponent)
    mantissa = paddle.where((mantissa <= -1),
                            paddle.divide(mantissa,
                                          2**paddle.ones_like(exponent)),
                            mantissa)
    mantissa = paddle.where((mantissa >= -1),
                            paddle.divide(mantissa,
                                          2**paddle.ones_like(exponent)),
                            mantissa)

    mantissa = paddle.where((x < 0), mantissa * -1, mantissa)
    return mantissa, exponent


class TestFrexpAPI(unittest.TestCase):

    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-3, 3, [10, 12]).astype('float32')
        self.place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    # 静态图单测
    def test_static_api(self):
        # 开启静态图模式
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            input_data = paddle.fluid.data('X', [10, 12])
            out2 = math.frexp(input_data)
            # out2 = frexp(input_data)
            # 计算静态图结果
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out2])
        out1 = np.frexp(self.x_np)
        # 对比静态图与 numpy 实现函数计算结果是否相同
        np.testing.assert_allclose(out1, res, rtol=1e-05)

    # 动态图单测
    def test_dygraph_api(self):
        input_num = paddle.to_tensor(self.x_np)
        # 关闭静态图模式
        paddle.disable_static(self.place)
        # 测试动态图 tensor.frexp 和 paddle.tensor.math.frexp 计算结果
        out1 = np.frexp(self.x_np)
        out2 = math.frexp(input_num)
        # out2 = frexp(input_num)
        np.testing.assert_allclose(out1, out2, rtol=1e-05)

        out1 = np.frexp(self.x_np)
        out2 = input_num.frexp()
        # out2 = frexp(input_num)
        np.testing.assert_allclose(out1, out2, rtol=1e-05)
        paddle.enable_static()


if __name__ == "__main__":
    a = TestFrexpAPI()
    a.setUp()
    a.test_dygraph_api()
    a.test_static_api()
