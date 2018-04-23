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

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.framework as framework
import paddle.fluid.optimizer as optimizer
from paddle.fluid.backward import calc_gradient


class TestCalcGradient(unittest.TestCase):
    def test_calc_gradient(self):
        x = layers.create_parameter(dtype="float32", shape=[5, 10])
        y = layers.create_parameter(dtype="float32", shape=[10, 8])
        mul_out = layers.mul(x=x, y=y)
        mean_out = layers.mean(mul_out)
        a = calc_gradient(mean_out, mul_out)
        b = calc_gradient(mean_out, x)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        exe.run(fluid.default_main_program(), feed={}, fetch_list=[a, b])


if __name__ == "__main__":
    unittest.main()
