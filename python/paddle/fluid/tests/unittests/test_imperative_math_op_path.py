# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import numpy as np

import unittest


class TestImperativeMathOp(unittest.TestCase):
    def test_mul(self):
        t = np.sqrt(2.0 * np.pi)
        fluid.enable_dygraph()
        x = fluid.layers.ones((2, 2), dtype="float32")
        y = t * x

        self.assertTrue(
            np.allclose(
                y.numpy(),
                t * np.ones(
                    (2, 2), dtype="float32"),
                rtol=1e-05,
                atol=0.0))


if __name__ == '__main__':
    unittest.main()
