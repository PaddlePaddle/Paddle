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

import numpy as np
from paddle import fluid, nn
import paddle.fluid.dygraph as dg
import paddle.fluid.initializer as I
import paddle.nn.functional as F
import unittest


class RowConvTestCase(unittest.TestCase):
    def __init__(self,
                 methodName='runTest',
                 batch_size=4,
                 num_channels=8,
                 time_steps=12,
                 context_size=3,
                 act=None,
                 dtype="float32"):
        super(RowConvTestCase, self).__init__(methodName=methodName)
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.time_steps = time_steps
        self.context_size = context_size
        self.act = act
        self.dtype = dtype

    def setUp(self):
        input_shape = (self.batch_size, self.time_steps, self.num_channels)
        self.input = np.random.uniform(size=input_shape).astype(self.dtype)
        self.weight_shape = weight_shape = (self.context_size + 1,
                                            self.num_channels)
        self.weight = np.random.uniform(size=weight_shape).astype(self.dtype)

    def fluid_layer(self, place):
        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                x = fluid.data(
                    "input", [-1, -1, self.num_channels], dtype=self.dtype)
                y = fluid.layers.row_conv(
                    x,
                    self.context_size,
                    param_attr=I.NumpyArrayInitializer(self.weight),
                    act=self.act)
        exe = fluid.Executor(place)
        exe.run(start)
        y_np, = exe.run(main, feed={"input": self.input}, fetch_list=[y])
        return y_np

    def runTest(self):
        place = fluid.CPUPlace()
        self._test_equivalence(place)

        if fluid.core.is_compiled_with_cuda():
            palce = fluid.CUDAPlace(0)
            self._test_equivalence(place)


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    suite.addTest(RowConvTestCase(methodName="runTest"))
    suite.addTest(RowConvTestCase(methodName="runTest", act="sigmoid"))
    suite.addTest(
        RowConvTestCase(
            methodName="runTest", context_size=5, act="sigmoid"))
    return suite


if __name__ == "__main__":
    unittest.main()
