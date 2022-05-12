#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
from paddle.fluid.framework import _test_eager_guard


class TestComplexTransposeLayer(unittest.TestCase):
    def setUp(self):
        self._dtypes = ["float32", "float64"]
        self._places = [paddle.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def test_transpose_by_complex_api(self):
        for dtype in self._dtypes:
            data = np.random.random(
                (2, 3, 4, 5)).astype(dtype) + 1J * np.random.random(
                    (2, 3, 4, 5)).astype(dtype)
            perm = [3, 2, 0, 1]
            np_trans = np.transpose(data, perm)
            for place in self._places:
                with dg.guard(place):
                    var = dg.to_variable(data)
                    trans = paddle.transpose(var, perm=perm)
                self.assertTrue(np.allclose(trans.numpy(), np_trans))

    def test_eager(self):
        with _test_eager_guard():
            self.test_transpose_by_complex_api()


if __name__ == '__main__':
    unittest.main()
