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


class TestComplexTransposeLayer(unittest.TestCase):
    def setUp(self):
        self._places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            self._places.append(fluid.CUDAPlace(0))

    def test_identity(self):
        data = np.random.random(
            (2, 3, 4, 5)).astype("float32") + 1J * np.random.random(
                (2, 3, 4, 5)).astype("float32")
        perm = [3, 2, 0, 1]
        np_trans = np.transpose(data, perm)
        for place in self._places:
            with dg.guard(place):
                var = dg.to_variable(data)
                trans = paddle.complex.transpose(var, perm=perm)
        self.assertTrue(np.allclose(trans.numpy(), np_trans))


if __name__ == '__main__':
    unittest.main()
