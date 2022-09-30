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
import unittest
import numpy as np
import paddle
import paddle.fluid


class TestFrexpAPI(unittest.TestCase):

    def setUp(self):
        np.random.seed(1024)
        self.rtol = 1e-5
        self.atol = 1e-8
        self.place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()
        self.set_input()

    def set_input(self):
        self.x_np = np.random.uniform(-3, 3, [10, 12]).astype('float32')

    # 静态图单测
    def test_static_api(self):
        # 开启静态图模式
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            input_data = paddle.fluid.data('X', self.x_np.shape,
                                           self.x_np.dtype)
            out = paddle.frexp(input_data)
            # 计算静态图结果
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])

        out_ref = np.frexp(self.x_np)
        # 对比静态图与 numpy 实现函数计算结果是否相同
        for n, p in zip(out_ref, res):
            np.testing.assert_allclose(n, p, rtol=self.rtol, atol=self.atol)

    # 动态图单测
    def test_dygraph_api(self):
        # 关闭静态图模式
        paddle.disable_static(self.place)
        input_num = paddle.to_tensor(self.x_np)
        # 测试动态图 tensor.frexp 和 paddle.tensor.math.frexp 计算结果
        out1 = np.frexp(self.x_np)
        out2 = paddle.frexp(input_num)
        np.testing.assert_allclose(out1, out2, rtol=1e-05)

        out1 = np.frexp(self.x_np)
        out2 = input_num.frexp()
        np.testing.assert_allclose(out1, out2, rtol=1e-05)
        paddle.enable_static()


class TestSplitsFloat32Case1(TestFrexpAPI):
    """
        Test num_or_sections which is an integer and data type is float32.
    """

    def set_input(self):
        self.x_np = np.random.uniform(-1, 1, [4, 5, 2]).astype('float32')


class TestSplitsFloat64Case1(TestFrexpAPI):
    """
        Test num_or_sections which is an integer and data type is float64.
    """

    def set_input(self):
        self.x_np = np.random.uniform(-3, 3, [10, 12]).astype('float64')


class TestSplitsFloat64Case2(TestFrexpAPI):
    """
        Test num_or_sections which is an integer and data type is float64.
    """

    def set_input(self):
        self.x_np = np.random.uniform(-1, 1, [4, 5, 2]).astype('float64')


if __name__ == "__main__":
    unittest.main()
