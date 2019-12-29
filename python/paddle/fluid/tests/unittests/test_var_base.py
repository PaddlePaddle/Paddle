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

from __future__ import print_function

import unittest
from paddle.fluid.framework import default_main_program, Program, convert_np_dtype_to_dtype_, in_dygraph_mode
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
import numpy as np


class TestVarBase(unittest.TestCase):
    def setUp(self):
        self.shape = [512, 1234]
        self.dtype = np.float32
        self.array = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)

    def test_to_variable(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array, name="abc")
            self.assertTrue(np.array_equal(var.numpy(), self.array))
            self.assertEqual(var.name, 'abc')
            # default value
            self.assertEqual(var.persistable, False)
            self.assertEqual(var.stop_gradient, True)
            self.assertEqual(var.shape, self.shape)
            self.assertEqual(var.dtype, core.VarDesc.VarType.FP32)
            self.assertEqual(var.type, core.VarDesc.VarType.LOD_TENSOR)

    def test_write_property(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array)

            self.assertEqual(var.name, 'generated_var_0')
            var.name = 'test'
            self.assertEqual(var.name, 'test')

            self.assertEqual(var.persistable, False)
            var.persistable = True
            self.assertEqual(var.persistable, True)

            self.assertEqual(var.stop_gradient, True)
            var.stop_gradient = False
            self.assertEqual(var.stop_gradient, False)

    # test some patched methods
    def test_set_value(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array)
            tmp1 = np.random.uniform(0.1, 1, [2, 2, 3]).astype(self.dtype)
            self.assertRaises(AssertionError, var.set_value, tmp1)

            tmp2 = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
            var.set_value(tmp2)
            self.assertTrue(np.array_equal(var.numpy(), tmp2))

    def test_to_string(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array)
            self.assertTrue(isinstance(str(var.to_string(True)), str))

    def test_backward(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array)
            var.stop_gradient = False
            loss = fluid.layers.relu(var)
            loss.backward()
            grad_var = var._grad_ivar()
            self.assertEqual(grad_var.shape, self.shape)

    def test_gradient(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array)
            var.stop_gradient = False
            loss = fluid.layers.relu(var)
            loss.backward()
            grad_var = var.gradient()
            self.assertEqual(grad_var.shape, self.array.shape)

    def test_block(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array)
            self.assertEqual(var.block,
                             fluid.default_main_program().global_block())

    def test_slice(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array)
            self.assertTrue(np.array_equal(var[1, :].numpy(), self.array[1, :]))
            self.assertTrue(np.array_equal(var[::-1].numpy(), self.array[::-1]))

    def test_var_base_to_np(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array)
            self.assertTrue(
                np.array_equal(var.numpy(),
                               fluid.framework._var_base_to_np(var)))


if __name__ == '__main__':
    unittest.main()
