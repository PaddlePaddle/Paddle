#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import numpy as np

class TestRetainGrad(unittest.TestCase):
    def setUp(self):
        # 修改：优先使用GPU，没有GPU则使用CPU
        if paddle.is_compiled_with_cuda():
            paddle.set_device('gpu')
        else:
            paddle.set_device('cpu')
        self.x = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
        self.y = self.x + self.x

    def test_retain_grad(self):
        """Test new API: retain_grad()"""
        self.y.retain_grad()  # New API
        loss = self.y.sum()
        loss.backward()
        np.testing.assert_array_equal(self.y.grad.numpy(), [2.0, 2.0, 2.0])
        print("✅ retain_grad() test passed")

    def test_retain_grads(self):
        """Test historical alias: retain_grads()"""
        self.y.retain_grads()  # Historical alias
        loss = self.y.sum()
        loss.backward()
        np.testing.assert_array_equal(self.y.grad.numpy(), [2.0, 2.0, 2.0])
        print("✅ retain_grads() test passed")

    def test_both_methods(self):
        """Test using both new and old APIs simultaneously"""
        # First use new API
        self.y.retain_grad()
        # Then use alias
        self.y.retain_grads()
        loss = self.y.sum()
        loss.backward()
        np.testing.assert_array_equal(self.y.grad.numpy(), [2.0, 2.0, 2.0])
        print("✅ Using both new and old APIs test passed")

if __name__ == '__main__':
    unittest.main()