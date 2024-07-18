#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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


class TestExecutor(unittest.TestCase):
    def test_mul(self):
        # 启用动态图模式
        paddle.disable_static()

        # 定义输入数据
        a_np = np.random.random((100, 784)).astype('float32')
        b_np = np.random.random((784, 100)).astype('float32')

        # 将 numpy 数组转换为 paddle 的 Tensor
        a = paddle.to_tensor(a_np)
        b = paddle.to_tensor(b_np)

        # 执行矩阵乘法
        out = paddle.matmul(a, b)

        # 验证结果
        self.assertEqual(tuple(out.shape), (100, 100))  # 转换为元组进行比较
        np.testing.assert_allclose(out.numpy(), np.dot(a_np, b_np), rtol=1e-05)

        # 重新启用静态图模式（如果需要）
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
