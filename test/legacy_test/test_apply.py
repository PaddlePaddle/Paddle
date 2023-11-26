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

import paddle


class TestTensorApplyAPI(unittest.TestCase):
    def setUp(self):
        self.x = paddle.to_tensor([1, 2, 3, 4, 5])

    def test_dygraph(self):
        def f(x):
            return 3 * x + 2

        y = self.x.apply(f)
        np.testing.assert_allclose(f(self.x).numpy(), y.numpy(), rtol=1e-05)

    def test_error(self):
        self.x.stop_gradient = False

        def f(x):
            x.apply_(x)

        self.assertRaises(RuntimeError, f, self.x)


if __name__ == "__main__":
    unittest.main()
