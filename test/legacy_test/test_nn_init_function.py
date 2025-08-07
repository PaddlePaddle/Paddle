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

import math
import random
import unittest

import numpy as np
from scipy import stats

import paddle
from paddle import nn


def get_uniform_min_and_max(weight):
    min_value = np.min(weight)
    max_value = np.max(weight)
    return min_value, max_value


class TestKaimingUniform(unittest.TestCase):
    def _test_kaiming_uniform_common(self, tensor):
        init = paddle.nn.init.kaiming_uniform_
        init(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu")
        init(tensor, a=-0.2, mode="fan_out", nonlinearity="leaky_relu")
        init(tensor, a=0, mode="fan_in", nonlinearity="relu")
        init(tensor, a=0, mode="fan_out", nonlinearity="relu")

    def test_kaiming_uniform_linear(self):
        linear = nn.Linear(40, 20)
        self._test_kaiming_uniform_common(linear.weight)

    def _create_random_nd_tensor(self, dims, size_min, size_max):
        size = [random.randint(size_min, size_max) for _ in range(dims)]
        tensor = paddle.zeros(size)
        return tensor

    def _is_uniform(self, tensor, a, b):
        samples = tensor.view([-1]).tolist()
        p_value = stats.kstest(samples, "uniform", args=(a, (b - a)))[1]
        return p_value > 0.0001

    def _random_float(self, a, b):
        return (b - a) * random.random() + a

    def test_kaiming_uniform(self):
        for use_a in [True, False]:
            for dims in [2, 4]:
                for mode in ["fan_in", "fan_out"]:
                    input_tensor = self._create_random_nd_tensor(
                        dims, size_min=20, size_max=25
                    )
                    if use_a:
                        a = self._random_float(0.1, 2)
                        paddle.nn.init.kaiming_uniform_(
                            input_tensor, a=a, mode=mode
                        )
                    else:
                        a = 0
                        paddle.nn.init.kaiming_uniform_(input_tensor, mode=mode)

                    if dims == 2:
                        # This is the case for simple matrix multiply
                        fan_in = input_tensor.shape[0]
                        fan_out = input_tensor.shape[1]
                    else:
                        fan_in = input_tensor.shape[1]
                        fan_out = input_tensor.shape[0]

                    if input_tensor.dim() > 2:
                        fan_in *= input_tensor[0, 0].numel()
                        fan_out *= input_tensor[0, 0].numel()

                    if mode == "fan_in":
                        n = fan_in
                    else:
                        n = fan_out

                    expected_std = math.sqrt(2.0 / ((1 + a**2) * n))
                    bounds = expected_std * math.sqrt(3.0)
                    assert self._is_uniform(input_tensor, -bounds, bounds)


if __name__ == '__main__':
    unittest.main()
