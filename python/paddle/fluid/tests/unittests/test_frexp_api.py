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

# def frexp(x):
#     if x.dtype not in [paddle.float32, paddle.float64]:
#         raise TypeError(
#             "The data type of input must be one of ['float32', 'float64'], but got {}"
#             .format(x.dtype))
#     input_x = paddle.abs(x)
#     exponent = paddle.floor(paddle.log2(input_x))
#     exponent = paddle.where(paddle.isinf(exponent),
#                             paddle.full_like(exponent, 0), exponent)
#
#     # 0填充
#     mantissa = paddle.divide(input_x, 2**exponent)
#     # 计算exponent
#     exponent = paddle.where((mantissa <= -1),
#                             paddle.add(exponent, paddle.ones_like(exponent)),
#                             exponent)
#     exponent = paddle.where((mantissa >= 1),
#                             paddle.add(exponent, paddle.ones_like(exponent)),
#                             exponent)
#     mantissa = paddle.where((mantissa <= -1),
#                             paddle.divide(mantissa,
#                                           2**paddle.ones_like(exponent)),
#                             mantissa)
#     mantissa = paddle.where((mantissa >= -1),
#                             paddle.divide(mantissa,
#                                           2**paddle.ones_like(exponent)),
#                             mantissa)
#
#     mantissa = paddle.where((x < 0), mantissa * -1, mantissa)
#     return mantissa, exponent


class TestFrexpAPI(unittest.TestCase):

    def setUp(self):
        np.random.seed(1024)
        self.rtol = 1e-5
        self.atol = 1e-8
        self.place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()
        self.set_input()

    def set_input(self):
        self.x_np_1 = np.random.uniform(-3, 3, [10, 12]).astype('float32')
        self.x_np_2 = np.random.uniform(-1, 1, [4, 5, 2]).astype('float32')

    # 静态图单测
    def test_static_api(self):
        # 开启静态图模式
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            input_data_1 = paddle.fluid.data('X', self.x_np_1.shape,
                                             self.x_np_1.dtype)
            out1 = math.frexp(input_data_1)
            # out1 = frexp(input_data_1)
            # 计算静态图结果
            exe = paddle.static.Executor(self.place)
            res_1 = exe.run(feed={'X': self.x_np_1}, fetch_list=[out1])

        with paddle.static.program_guard(paddle.static.Program()):
            input_data_2 = paddle.fluid.data('X', self.x_np_2.shape,
                                             self.x_np_2.dtype)
            out2 = math.frexp(input_data_2)
            # out2 = frexp(input_data_2)
            # 计算静态图结果
            exe = paddle.static.Executor(self.place)
            res_2 = exe.run(feed={'X': self.x_np_2}, fetch_list=[out2])

        out_ref_1 = np.frexp(self.x_np_1)
        # 对比静态图与 numpy 实现函数计算结果是否相同
        for n, p in zip(out_ref_1, res_1):
            np.testing.assert_allclose(n, p, rtol=self.rtol, atol=self.atol)

        out_ref_2 = np.frexp(self.x_np_2)
        # 对比静态图与 numpy 实现函数计算结果是否相同
        for n, p in zip(out_ref_2, res_2):
            np.testing.assert_allclose(n, p, rtol=self.rtol, atol=self.atol)

    # 动态图单测
    def test_dygraph_api(self):
        input_num = paddle.to_tensor(self.x_np_1)
        # 关闭静态图模式
        paddle.disable_static(self.place)
        # 测试动态图 tensor.frexp 和 paddle.tensor.math.frexp 计算结果
        out1 = np.frexp(self.x_np_1)
        out2 = math.frexp(input_num)
        # out2 = frexp(input_num)
        np.testing.assert_allclose(out1, out2, rtol=1e-05)

        out1 = np.frexp(self.x_np_1)
        out2 = input_num.frexp()
        np.testing.assert_allclose(out1, out2, rtol=1e-05)

        input_num = paddle.to_tensor(self.x_np_2)
        out1 = np.frexp(self.x_np_2)
        out2 = math.frexp(input_num)
        np.testing.assert_allclose(out1, out2, rtol=1e-05)

        out1 = np.frexp(self.x_np_2)
        out2 = input_num.frexp()
        np.testing.assert_allclose(out1, out2, rtol=1e-05)
        paddle.enable_static()


class TestSplitsFloat32(TestFrexpAPI):
    """
        Test num_or_sections which is an integer and data type is float32.
    """

    def set_input(self):
        self.x_np_1 = np.random.uniform(-3, 3, [10, 12]).astype('float64')


if __name__ == "__main__":
    # unittest.main()
    a = TestFrexpAPI()
    a.setUp()
    a.test_dygraph_api()
    a.test_static_api()
