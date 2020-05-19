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
import paddle.fluid.core as core
import paddle.nn as nn
import paddle.nn.functional as functional


def stable_softmax(x):
    shiftx = (x - np.max(x))
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def ref_log_softmax(x, axis=None, dtype=None):
    x_t = x.copy()
    if dtype is not None:
        x_t = x_t.astype(dtype)
    if axis is None:
        axis = -1
    out = np.apply_along_axis(stable_softmax, axis, x_t)
    return np.log(out)


class TestNNLogSoftmaxAPI(unittest.TestCase):
    def setUp(self):
        self.init_data()

    def init_data(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)

    def check_api(self, place=fluid.CPUPlace(), axis=None):
        ref_out = ref_log_softmax(self.x, axis)

        main_program = fluid.Program()
        mylogsoftmax = nn.LogSoftmax(axis)
        with fluid.program_guard(main_program):
            x = fluid.data(name='x', shape=self.x_shape)
            y = mylogsoftmax(x)
        exe = fluid.Executor(place)
        out = exe.run(main_program, feed={'x': self.x}, fetch_list=[y])
        self.assertTrue(np.allclose(out[0], ref_out))

        with fluid.dygraph.guard(place):
            x = fluid.dygraph.to_variable(self.x)
            y = mylogsoftmax(x)
        self.assertTrue(np.allclose(y.numpy(), ref_out))

    def test_check_api(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            for axis in [None, 2]:
                self.check_api(place, axis)


class TestNNFunctionalLogSoftmaxAPI(unittest.TestCase):
    def setUp(self):
        self.init_data()

    def init_data(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)

    def check_api(self, place=fluid.CPUPlace(), axis=None, dtype=None):
        ref_out = ref_log_softmax(self.x, axis, dtype)
        main_program = fluid.Program()
        mylogsoftmax = nn.LogSoftmax(axis)
        with fluid.program_guard(main_program):
            x = fluid.data(name='x', shape=self.x_shape)
            y = functional.log_softmax(x, axis, dtype)
        exe = fluid.Executor(place)
        out = exe.run(main_program, feed={'x': self.x}, fetch_list=[y])
        self.assertTrue(np.allclose(out[0], ref_out))

        with fluid.dygraph.guard(place):
            x = fluid.dygraph.to_variable(self.x)
            y = functional.log_softmax(x, axis, dtype)
        self.assertTrue(np.allclose(y.numpy(), ref_out))

    def test_check_api(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            self.check_api(place, None, None)
            self.check_api(place, None, np.float64)


if __name__ == "__main__":
    unittest.main()
