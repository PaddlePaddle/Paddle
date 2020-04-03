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
import numpy as np
import paddle.fluid as fluid
import paddle.nn as nn
import paddle.nn.functional as functional
import gradient_checker

from decorator_helper import prog_scope


class TestNNLogSoftmaxAPI(unittest.TestCase):
    @prog_scope()
    def func(self, place, axis=None):
        shape = [2, 3, 4, 5]
        eps = 0.005

        x = fluid.data('x', shape)
        x.persistable = True
        my_log_softmax = nn.LogSoftmax(axis)
        y = my_log_softmax(x)

        x_arr = np.random.uniform(-1, 1, shape).astype(np.float32)
        gradient_checker.grad_check([x], y, x_init=x_arr, place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            for axis in [None, 2]:
                self.func(place, axis)


class TestNNFunctionalLogSoftmaxAPI(unittest.TestCase):
    @prog_scope()
    def func(self, place, axis=None, dtype=None):
        shape = [2, 3, 4, 5]
        eps = 0.005

        x = fluid.data('x', shape)
        x.persistable = True
        y = functional.log_softmax(x, axis, dtype)

        x_arr = np.random.uniform(-1, 1, shape).astype(np.float32)
        gradient_checker.grad_check([x], y, x_init=x_arr, place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            self.func(place, None, np.float64)


if __name__ == "__main__":
    unittest.main()
