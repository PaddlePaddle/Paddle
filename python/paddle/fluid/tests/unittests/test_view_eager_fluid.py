#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.framework import _test_eager_guard


# NOTE(pangyoki): Tensor View Strategy.
# For view op, a new output Tensor will be created, and this output will
# reuse the input Tensor's allocation.
# View APIs include: `squeeze`, `unsqueeze`, `reshape`, `flatten`
class TestDygraphViewReuseAllocation(unittest.TestCase):
    def setUp(self):
        self.init_shape()

    def init_shape(self):
        self.input_shape = [2, 3, 1]
        self.output_shape = [2, 3]

    def view_api_processing(self, var):
        return paddle.squeeze(var)

    def test_view_api(self):
        with _test_eager_guard():
            var = paddle.rand(self.input_shape)
            view_var = self.view_api_processing(var)
            view_var.exp_()
            self.assertEqual(var.shape, self.input_shape)
            self.assertEqual(view_var.shape, self.output_shape)

            var_numpy = var.numpy().reshape(self.output_shape)
            view_var_numpy = view_var.numpy()
            self.assertTrue(np.array_equal(var_numpy, view_var_numpy))

    def test_forward_version(self):
        with _test_eager_guard():
            var = paddle.rand(self.input_shape)
            self.assertEqual(var.inplace_version, 0)
            view_var = self.view_api_processing(var)
            self.assertEqual(view_var.inplace_version, 0)

            var.exp_()
            self.assertEqual(var.inplace_version, 1)
            self.assertEqual(view_var.inplace_version, 1)

            view_var_2 = self.view_api_processing(var)
            self.assertEqual(view_var_2.inplace_version, 1)

            var.exp_()
            self.assertEqual(view_var.inplace_version, 2)
            self.assertEqual(view_var_2.inplace_version, 2)

    def test_backward_error(self):
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.fluid.dygraph.guard():
            with _test_eager_guard():
                var_a = paddle.ones(shape=self.input_shape, dtype="float32")
                var_a.stop_gradient = False

                var_b = var_a**2

                # Here, the gradient computation will use the value of var_b
                var_c = var_b**2
                view_var_b = self.view_api_processing(var_b)
                view_var_b.exp_()  # var_b is modified inplace

                loss = paddle.nn.functional.relu(var_c)
                with self.assertRaisesRegexp(
                        RuntimeError,
                        "received current_inplace_version:{} != inplace_version_snapshot_:{}".
                        format(1, 0)):
                    loss.backward()


class TestUnsqueezeDygraphViewReuseAllocation(TestDygraphViewReuseAllocation):
    def init_shape(self):
        self.input_shape = [2, 3]
        self.output_shape = [2, 3, 1]

    def view_api_processing(self, var):
        return paddle.unsqueeze(var, -1)


class TestReshapeDygraphViewReuseAllocation(TestDygraphViewReuseAllocation):
    def init_shape(self):
        self.input_shape = [3, 4]
        self.output_shape = [2, 2, 3]

    def view_api_processing(self, var):
        return paddle.reshape(var, [2, 2, 3])


class TestFlattenDygraphViewReuseAllocation(TestDygraphViewReuseAllocation):
    def init_shape(self):
        self.input_shape = [3, 4]
        self.output_shape = [12]

    def view_api_processing(self, var):
        return paddle.flatten(var)


if __name__ == "__main__":
    unittest.main()
