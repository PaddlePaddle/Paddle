#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard
from op_test import OpTest


class TestHistogramOpAPI(unittest.TestCase):
    """Test histogram api."""

    def test_static_graph(self):
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            inputs = fluid.data(name='input', dtype='int64', shape=[2, 3])
            output = paddle.histogram(inputs, bins=5, min=1, max=5)
            place = fluid.CPUPlace()
            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup_program)
            img = np.array([[2, 4, 2], [2, 5, 4]]).astype(np.int64)
            res = exe.run(train_program,
                          feed={'input': img},
                          fetch_list=[output])
            actual = np.array(res[0])
            expected = np.array([0, 3, 0, 2, 1]).astype(np.int64)
            self.assertTrue(
                (actual == expected).all(),
                msg='histogram output is wrong, out =' + str(actual))

    def test_dygraph(self):
        with fluid.dygraph.guard():
            inputs_np = np.array([[2, 4, 2], [2, 5, 4]]).astype(np.int64)
            inputs = fluid.dygraph.to_variable(inputs_np)
            actual = paddle.histogram(inputs, bins=5, min=1, max=5)
            expected = np.array([0, 3, 0, 2, 1]).astype(np.int64)
            self.assertTrue(
                (actual.numpy() == expected).all(),
                msg='histogram output is wrong, out =' + str(actual.numpy()))


class TestHistogramOp(OpTest):
    def setUp(self):
        self.op_type = "histogram"
        self.init_test_case()
        np_input = np.random.randint(
            low=0, high=20, size=self.in_shape, dtype=np.int64)
        self.inputs = {"X": np_input}
        self.init_attrs()
        Out, _ = np.histogram(
            np_input, bins=self.bins, range=(self.min, self.max))
        self.outputs = {"Out": Out.astype(np.int64)}

    def init_test_case(self):
        self.in_shape = (10, 12)
        self.bins = 5
        self.min = 1
        self.max = 5

    def init_attrs(self):
        self.attrs = {"bins": self.bins, "min": self.min, "max": self.max}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
