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

import paddle
import paddle.fluid as fluid
from paddle.framework import seed
from paddle.fluid.framework import Program, default_main_program, default_startup_program
import numpy as np


class TestManualSeed(unittest.TestCase):

    def test_seed(self):
        fluid.enable_dygraph()

        gen = paddle.seed(12312321111)
        x = fluid.layers.gaussian_random([10], dtype="float32")
        st1 = gen.get_state()
        x1 = fluid.layers.gaussian_random([10], dtype="float32")
        gen.set_state(st1)
        x2 = fluid.layers.gaussian_random([10], dtype="float32")
        gen.manual_seed(12312321111)
        x3 = fluid.layers.gaussian_random([10], dtype="float32")
        x_np = x.numpy()
        x1_np = x1.numpy()
        x2_np = x2.numpy()
        x3_np = x3.numpy()

        if not fluid.core.is_compiled_with_cuda():
            np.testing.assert_allclose(x1_np, x2_np, rtol=1e-05)
            np.testing.assert_allclose(x_np, x3_np, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
