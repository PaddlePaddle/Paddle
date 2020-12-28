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

from op_test import OpTest
import paddle


class TestDygraphInplaceReuseAllocation(unittest.TestCase):
    def reuse_allocation_api_processing(self, var):
        return paddle.squeeze(var)

    def get_numpy_result(self, var_numpy):
        return var_numpy.squeeze()

    def test_inplace_reuse_allocation(self):
        var = paddle.rand([2, 3, 1])
        inplace_var = self.reuse_allocation_api_processing(var)
        inplace_var[0] = 2.
        self.assertNotEqual(var.shape, inplace_var.shape)

        var_numpy = self.get_numpy_result(var.numpy())
        inplace_var_numpy = inplace_var.numpy()
        self.assertTrue(np.array_equal(var_numpy, inplace_var_numpy))

    def test_forward_version(self):
        var = paddle.rand([2, 3, 1])
        self.assertEqual(var.inplace_version, 0)
        inplace_var = self.reuse_allocation_api_processing(var)
        self.assertEqual(inplace_var.inplace_version, 0)

        var[0] = 2.
        self.assertEqual(var.inplace_version, 1)
        self.assertEqual(inplace_var.inplace_version, 1)

        inplace_var_2 = self.reuse_allocation_api_processing(var)
        self.assertEqual(inplace_var_2.inplace_version, 1)

        var[0] = 3.
        self.assertEqual(inplace_var.inplace_version, 2)
        self.assertEqual(inplace_var_2.inplace_version, 2)

    def test_backward_error(self):
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.fluid.dygraph.guard():
            var_a = paddle.ones(shape=[2, 3, 1], dtype="float32")
            var_a.stop_gradient = False

            var_b = var_a**2

            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            inplace_var_b = self.reuse_allocation_api_processing(var_b)
            inplace_var_b[0] = 2.  # var_b is modified inplace

            loss = paddle.nn.functional.relu(var_c)
            with self.assertRaisesRegexp(
                    RuntimeError,
                    "received tensor_version:{} != wrapper_version_snapshot:{}".
                    format(1, 0)):
                loss.backward()


class TestUnsqueezeDygraphInplaceReuseAllocation(
        TestDygraphInplaceReuseAllocation):
    def reuse_allocation_api_processing(self, var):
        return paddle.unsqueeze(var, -1)

    def get_numpy_result(self, var_numpy):
        return var_numpy.reshape([2, 3, 1, 1])


class TestReshapeDygraphInplaceReuseAllocation(
        TestDygraphInplaceReuseAllocation):
    def reuse_allocation_api_processing(self, var):
        return paddle.reshape(var, [6, 1])

    def get_numpy_result(self, var_numpy):
        return var_numpy.reshape([6, 1])


class TestFlattenDygraphInplaceReuseAllocation(
        TestDygraphInplaceReuseAllocation):
    def reuse_allocation_api_processing(self, var):
        return paddle.flatten(var)

    def get_numpy_result(self, var_numpy):
        return var_numpy.reshape([6])


if __name__ == "__main__":
    unittest.main()
