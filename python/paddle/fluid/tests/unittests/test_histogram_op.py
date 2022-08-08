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
from paddle.fluid.framework import _test_eager_guard


class TestHistogramOpAPI(unittest.TestCase):
    """Test histogram api."""

    def test_static_graph(self):
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            inputs = fluid.data(name='input', dtype='int64', shape=[2, 3])
            place = fluid.CPUPlace()
            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup_program)
            img = np.array([[2, 4, 2], [2, 5, 4]]).astype(np.int64)

            # test int bins
            output_int = paddle.histogram(inputs, bins=5, min=1, max=5)
            res_int = exe.run(train_program,
                              feed={'input': img},
                              fetch_list=[output_int])
            actual_int = np.array(res_int[0])
            expected_int = np.array([0, 3, 0, 2, 1]).astype(np.int64)
            self.assertTrue(
                (actual_int == expected_int).all(),
                msg='histogram output is wrong, out =' + str(actual_int))

            # test list bins
            output_list = paddle.histogram(inputs, bins=[1, 2, 3, 4, 5])
            res_list = exe.run(train_program,
                               feed={'input': img},
                               fetch_list=[output_list])
            actual_list = np.array(res_list[0])
            expected_list = np.array([0, 3, 0, 3]).astype(np.int64)
            self.assertTrue(
                (actual_list == expected_list).all(),
                msg='histogram output is wrong, out =' + str(actual_list))

            # test Tensor bins
            inputs = (fluid.data(name='input', dtype='int64', shape=[2, 3]),
                      fluid.data(name='bins', dtype='int64', shape=[5]))
            output_tensor = paddle.histogram(inputs[0], inputs[1])
            bins = np.array([1, 2, 3, 4, 5]).astype(np.int64)
            res_tensor = exe.run(train_program,
                                 feed={
                                     'input': img,
                                     'bins': bins
                                 },
                                 fetch_list=[output_tensor])
            actual_tensor = np.array(res_tensor[0])
            expected_tensor = np.array([0, 3, 0, 3]).astype(np.int64)
            self.assertTrue(
                (actual_tensor == expected_tensor).all(),
                msg='histogram output is wrong, out =' + str(actual_tensor))

    def test_dygraph(self):
        with fluid.dygraph.guard():
            inputs_np = np.array([[2, 4, 2], [2, 5, 4]]).astype(np.int64)
            inputs = fluid.dygraph.to_variable(inputs_np)

            actual_int = paddle.histogram(inputs, bins=5, min=1, max=5)
            expected_int = np.array([0, 3, 0, 2, 1]).astype(np.int64)
            self.assertTrue((actual_int.numpy() == expected_int).all(),
                            msg='histogram output is wrong, out =' +
                            str(actual_int.numpy()))

            actual_list = paddle.histogram(inputs, bins=[1, 2, 3, 4, 5])
            expected_list = np.array([0, 3, 0, 3]).astype(np.int64)
            self.assertTrue((actual_list.numpy() == expected_list).all(),
                            msg='histogram output is wrong, out =' +
                            str(actual_list.numpy()))

            actual_tensor = paddle.histogram(inputs,
                                             bins=paddle.to_tensor(
                                                 [1, 2, 3, 4, 5]),
                                             min=1,
                                             max=5)
            expected_tensor = np.array([0, 3, 0, 3]).astype(np.int64)
            self.assertTrue((actual_tensor.numpy() == expected_tensor).all(),
                            msg='histogram output is wrong, out =' +
                            str(actual_tensor.numpy()))

            with _test_eager_guard():
                inputs_np = np.array([[2, 4, 2], [2, 5, 4]]).astype(np.int64)
                inputs = paddle.to_tensor(inputs_np)

                actual_int = paddle.histogram(inputs, bins=5, min=1, max=5)
                self.assertTrue((actual_int.numpy() == expected_int).all(),
                                msg='histogram output is wrong, out =' +
                                str(actual_int.numpy()))
                actual_list = paddle.histogram(inputs, bins=[1, 2, 3, 4, 5])
                self.assertTrue((actual_list.numpy() == expected_list).all(),
                                msg='histogram output is wrong, out =' +
                                str(actual_list.numpy()))
                actual_tensor = paddle.histogram(inputs,
                                                 bins=paddle.to_tensor(
                                                     [1, 2, 3, 4, 5]))
                self.assertTrue(
                    (actual_tensor.numpy() == expected_tensor).all(),
                    msg='histogram output is wrong, out =' +
                    str(actual_tensor.numpy()))


class TestHistogramOpError(unittest.TestCase):
    """Test histogram op error."""

    def run_network(self, net_func):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            net_func()
            exe = fluid.Executor()
            exe.run(main_program)

    def test_bins_error(self):
        """Test bins should be int, list or Tensor. When bins is an int, it should be greater than or equal to 1.
        When bins is a list or a Tensor, it should be 1D, contain at least 2 elements and increase monotonically."""

        # When bins is an int, it should be greater than or equal to 1.
        def net_func_1():
            input_value = paddle.fluid.layers.fill_constant(shape=[3, 4],
                                                            dtype='float32',
                                                            value=3.0)
            paddle.histogram(input=input_value, bins=-1, min=1, max=5)

        # When bins is a list or tensor, it should increase monotonically
        def net_func_2():
            input_value = paddle.fluid.layers.fill_constant(shape=[3, 4],
                                                            dtype='float32',
                                                            value=3.0)
            paddle.histogram(input=input_value, bins=[1, 5, 3], min=1, max=5)

        with self.assertRaises(ValueError):
            self.run_network(net_func_1)

        with self.assertRaises(ValueError):
            self.run_network(net_func_2)

        with program_guard(Program()):
            input_value = paddle.fluid.layers.fill_constant(shape=[3, 4],
                                                            dtype='float32',
                                                            value=3.0)
            # bins type must be int, list or Tensor
            self.assertRaises(TypeError,
                              paddle.histogram,
                              input=input_value,
                              bins=3.8,
                              min=1,
                              max=5)
            # When bins is a list or a Tensor, the dtype should be 'int32', 'int64', 'float32', 'float64'
            self.assertRaises(TypeError,
                              paddle.histogram,
                              input=input_value,
                              bins=fluid.data(name='bins_bool',
                                              shape=[5],
                                              dtype='bool'),
                              min=1,
                              max=5)
            # When bins is a list or a Tensor, it should be 1D
            self.assertRaises(TypeError,
                              paddle.histogram,
                              input=input_value,
                              bins=[[1, 2], [3, 4]],
                              min=1,
                              max=5)
            # When bins is a list or a Tensor, it should contain at least 2 elements
            self.assertRaises(TypeError,
                              paddle.histogram,
                              input=input_value,
                              bins=[1],
                              min=1,
                              max=5)

    def test_min_max_error(self):
        """Test max must be larger or equal to min."""

        def net_func():
            input_value = paddle.fluid.layers.fill_constant(shape=[3, 4],
                                                            dtype='float32',
                                                            value=3.0)
            paddle.histogram(input=input_value, bins=1, min=5, max=1)

        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_min_max_range_error(self):
        """Test range of min, max is not finite"""

        def net_func():
            input_value = paddle.fluid.layers.fill_constant(shape=[3, 4],
                                                            dtype='float32',
                                                            value=3.0)
            paddle.histogram(input=input_value, bins=1, min=-np.inf, max=5)

        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_type_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError,
                              paddle.histogram,
                              1,
                              bins=5,
                              min=1,
                              max=5)
            # The input type must be 'int32', 'int64', 'float32', 'float64'
            x_bool = fluid.data(name='x_bool', shape=[4, 3], dtype='bool')
            self.assertRaises(TypeError,
                              paddle.histogram,
                              x_bool,
                              bins=5,
                              min=1,
                              max=5)


class TestHistogramOp(OpTest):

    def setUp(self):
        self.op_type = "histogram"
        self.init_test_case()
        np_input = np.random.uniform(low=0.0, high=20.0, size=self.in_shape)
        self.python_api = paddle.histogram
        self.inputs = {"X": np_input, "bins": self.bins}
        self.init_attrs()
        Out, _ = np.histogram(np_input,
                              bins=self.bins,
                              range=(self.min, self.max))
        self.outputs = {"Out": Out.astype(np.int64)}

    def init_test_case(self):
        self.in_shape = (10, 12)
        self.bins = np.asarray([1.0, 3.0, 5.0])
        self.min = 1
        self.max = 5

    def init_attrs(self):
        self.attrs = {"min": self.min, "max": self.max}

    def test_check_output(self):
        self.check_output(check_eager=True)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
