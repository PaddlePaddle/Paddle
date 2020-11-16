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

from __future__ import print_function

import unittest
import numpy as np

import paddle
import paddle.fluid.core as core
import functools

# class TestInplace(unittest.TestCase):
#
#     def setUp(self):
#         self.inplace_apis_with_same_varbase = [
#             self._assign, self._setitem]
#
#         self.inplace_apis_with_diff_varbase = [
#             self._detach
#         ]
#
#     def _assign(self, var):
#         input = paddle.ones(shape=[3])
#         return paddle.nn.functional.assign(input=input, output=var)
#
#     def _setitem(self, var):
#         var[1] = 3
#         return var
#
#     def _detach(self, var):
#         var.detach()
#         return var
#
#
#     def test_forward_version(self):
#         with paddle.fluid.dygraph.guard():
#             var = paddle.to_tensor(np.ones((4, 2, 3)).astype(np.float32))
#             self.assertEqual(var.inplace_version, 0)
#
#             inplace_func_num = len(self.inplace_apis_with_same_varbase)
#
#             for i, inplace_func in enumerate(self.inplace_apis_with_same_varbase):
#                 var = inplace_func(var)
#                 self.assertEqual(var.inplace_version, i+1)
#
#             for i, inplace_func in enumerate(self.inplace_apis_with_same_varbase):
#                 var = inplace_func(var)
#                 self.assertEqual(var.inplace_version, i+inplace_func_num+1)
#
#
#
#             # var[0] = 1.1
#             # self.assertEqual(var.inplace_version, 1)
#             #
#             # # paddle.nn.functional.assign(paddle.ones(shape=[3]), var)
#             # self._assign(var)
#             #
#             # # NOTE(liym27): assign(input, output) is an inplace operation for output.
#             # # There is inplace-related processing for api assign, var.inplace_version should be 2 not 1.
#             # self.assertEqual(var.inplace_version, 2)
#             #
#             # var[2] = 3
#             # self.assertEqual(var.inplace_version, 3)
#             #
#             # detach_var = var.detach()
#             #
#             # self.assertEqual(detach_var.inplace_version, 3)
#             #
#             # var[1] = 2
#             # self.assertEqual(detach_var.inplace_version, 4)
#             # self.assertEqual(var.inplace_version, 4)
#
#     # def test_forward_version(self):
#     #     with paddle.fluid.dygraph.guard():
#     #         var = paddle.to_tensor(np.ones((4, 2, 3)).astype(np.float32))
#     #         self.assertEqual(var.inplace_version, 0)
#     #
#     #         var[0] = 1.1
#     #         self.assertEqual(var.inplace_version, 1)
#     #
#     #         # paddle.nn.functional.assign(paddle.ones(shape=[3]), var)
#     #         self._assign(var)
#     #
#     #         # NOTE(liym27): assign(input, output) is an inplace operation for output.
#     #         # There is inplace-related processing for api assign, var.inplace_version should be 2 not 1.
#     #         self.assertEqual(var.inplace_version, 2)
#     #
#     #         var[2] = 3
#     #         self.assertEqual(var.inplace_version, 3)
#     #
#     #         detach_var = var.detach()
#     #
#     #         self.assertEqual(detach_var.inplace_version, 3)
#     #
#     #         var[1] = 2
#     #         self.assertEqual(detach_var.inplace_version, 4)
#     #         self.assertEqual(var.inplace_version, 4)
#
#     def test_backward_error(self):
#         # It raises an error because the inplace operator will result
#         # in incorrect gradient computation.
#         with paddle.fluid.dygraph.guard():
#             var_a = paddle.ones(shape=[4, 2, 3], dtype="float32")
#             var_a.stop_gradient = False
#
#             var_b = var_a**2
#
#             # Here, the gradient computation will use the value of var_b
#             var_c = var_b**2
#             detach_var_b = var_b.detach()
#             detach_var_b[1:2] = 3.3  # var_b is modified inplace after using it
#             # var_b[1:2] = 3.3  # var_b is modified inplace after using it
#
#             var_d = var_b**2
#
#             loss = paddle.nn.functional.relu(var_c + var_d)
#             with self.assertRaisesRegexp(
#                     RuntimeError,
#                     "received tensor_version:{} != wrapper_version_snapshot:{}".
#                     format(1, 0)):
#                 loss.backward()
#
#     def test_backward_success_1(self):
#         # var_b is modified inplace before using it, the inplace operator doesn't result
#         # in incorrect gradient computation.
#         with paddle.fluid.dygraph.guard():
#             var_a = paddle.ones(shape=[4, 2, 3], dtype="float32")
#             var_a.stop_gradient = False
#
#             var_b = var_a**2
#             var_b[1:2] = 3  # var_b is modified inplace before using it
#
#             # Here, the gradient computation will use the value of var_b
#             var_c = var_b**2
#             loss = var_c.sum()
#             loss.backward()
#
#     def test_backward_success_2(self):
#         # Although var_b is modified inplace after using it, it does not used in gradient computation.
#         # The inplace operator doesn't result in incorrect gradient computation.
#         with paddle.fluid.dygraph.guard():
#             var_a = paddle.ones(shape=[4, 2, 3], dtype="float32")
#             var_a.stop_gradient = False
#
#             var_b = var_a**2
#
#             var_b[1:2] = 3  # var_b is modified inplace before using it
#
#             var_c = var_b + var_b  # Here, the grad op of sum doesn't use the value of var_b
#             loss = var_c.sum()
#
#             var_b[1:2] = 3  # var_b is modified inplace after using it
#
#             loss.backward()


class TestInplaceSqueeze(unittest.TestCase):
    def test_forward_version(self):
        with paddle.fluid.dygraph.guard():
            var = paddle.to_tensor(np.ones((4, 2, 3)).astype(np.float32))
            self.assertEqual(var.inplace_version, 0)

            var[0] = 1.1
            self.assertEqual(var.inplace_version, 1)

            # paddle.nn.functional.assign(paddle.ones(shape=[3]), var)
            var = var.squeeze_()

            # NOTE(liym27): assign(input, output) is an inplace operation for output.
            # There is inplace-related processing for api assign, var.inplace_version should be 2 not 1.
            self.assertEqual(var.inplace_version, 2)

            var[2] = 3
            self.assertEqual(var.inplace_version, 3)

            detach_var = var.squeeze()

            self.assertEqual(detach_var.inplace_version, 3)

            var[1] = 2
            self.assertEqual(detach_var.inplace_version, 4)
            self.assertEqual(var.inplace_version, 4)

    def _test_backward_error(self):
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.fluid.dygraph.guard():
            var_a = paddle.ones(shape=[4, 2, 3], dtype="float32")
            var_a.stop_gradient = False

            var_b = var_a**2

            # Here, the gradient computation will use the value of var_b
            print("")
            var_c = var_b**2
            print("")
            var_d = var_b.squeeze_()

            loss = paddle.nn.functional.relu(var_c)
            loss.backward()
            # with self.assertRaisesRegexp(
            #         RuntimeError,
            #         "received tensor_version:{} != wrapper_version_snapshot:{}".
            #                 format(1, 0)):
            #     loss.backward()


if __name__ == '__main__':
    unittest.main()
