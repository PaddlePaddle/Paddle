# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


class TestHessian(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.shape = (4, 4)
        self.dtype = 'float32'
        self.np_dtype = np.float32
        self.numerical_delta = 1e-4
        self.rtol = 1e-3
        self.atol = 1e-3
        self.x = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.shape, dtype=self.dtype)

    def test_single_input(self):
        def func(x):
            return paddle.sum(paddle.matmul(x, x))

        self.x.stop_gradient = False
        hessian = paddle.autograd.hessian(func, self.x)
        print("hessian: ", hessian)


if __name__ == "__main__":
    unittest.main()
