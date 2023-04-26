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

import paddle.fluid.core as paddle_c
import unittest
import six


class TestScope(unittest.TestCase):
    def test_clear_vars(self):
        scope = paddle_c.Scope()
        var_names = ["var_a", "var_b", "var_c", "var_d"]
        for var_name in var_names:
            scope.var(var_name).get_tensor().set(
                [0, 1, 2, 3], paddle_c.CPUPlace()
            )

        local_var_names = scope.local_var_names()
        self.assertEqual(len(var_names), len(local_var_names))

        var_names_to_cleared = [var_names[0], var_names[1], var_names[2]]
        scope.clear(var_names_to_cleared)

        for var_name in local_var_names:
            actual_initialized_value = scope.var(var_name).is_initialized()
            expect_initialized_value = var_name not in var_names_to_cleared
            self.assertEqual(actual_initialized_value, expect_initialized_value)

    def test_create_destroy(self):
        scope = paddle_c.Scope()
        self.assertIsNotNone(scope)
        scope_with_parent = scope.new_scope()
        self.assertIsNotNone(scope_with_parent)

    def test_none_variable(self):
        scope = paddle_c.Scope()
        self.assertIsNone(scope.find_var("test"))

    def test_create_var_get_var(self):
        scope = paddle_c.Scope()
        var_a = scope.var("var_a")
        self.assertIsNotNone(var_a)
        self.assertIsNotNone(scope.find_var('var_a'))
        scope2 = scope.new_scope()
        self.assertIsNotNone(scope2.find_var('var_a'))

    def test_var_get_int(self):
        scope = paddle_c.Scope()
        var = scope.var("test_int")
        var.set_int(10)
        self.assertTrue(var.is_int())
        self.assertEqual(10, var.get_int())

    def test_scope_pool(self):
        scope = paddle_c.Scope()
        # Delete the scope.
        scope._remove_from_pool()
        with self.assertRaisesRegexp(
            Exception, "Deleting a nonexistent scope is not allowed*"
        ):
            # It is not allowed to delete a nonexistent scope.
            scope._remove_from_pool()

    def test_size(self):
        scope = paddle_c.Scope()
        var_a = scope.var("var_a")
        self.assertEqual(scope.size(), 1)
        self.assertIsNotNone(scope.find_var('var_a'))


if __name__ == '__main__':
    unittest.main()
