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

import paddle.fluid.core
import unittest
import six


class TestScope(unittest.TestCase):

    def test_create_destroy(self):
        paddle_c = paddle.fluid.core
        scope = paddle_c.Scope()
        self.assertIsNotNone(scope)
        scope_with_parent = scope.new_scope()
        self.assertIsNotNone(scope_with_parent)

    def test_none_variable(self):
        paddle_c = paddle.fluid.core
        scope = paddle_c.Scope()
        self.assertIsNone(scope.find_var("test"))

    def test_create_var_get_var(self):
        paddle_c = paddle.fluid.core
        scope = paddle_c.Scope()
        var_a = scope.var("var_a")
        self.assertIsNotNone(var_a)
        self.assertIsNotNone(scope.find_var('var_a'))
        scope2 = scope.new_scope()
        self.assertIsNotNone(scope2.find_var('var_a'))

    def test_var_get_int(self):
        paddle_c = paddle.fluid.core
        scope = paddle_c.Scope()
        var = scope.var("test_int")
        var.set_int(10)
        self.assertTrue(var.is_int())
        self.assertEqual(10, var.get_int())

    def test_scope_pool(self):
        paddle_c = paddle.fluid.core
        scope = paddle_c.Scope()
        # Delete the scope.
        scope._remove_from_pool()
        with self.assertRaisesRegexp(
                Exception, "Deleting a nonexistent scope is not allowed*"):
            # It is not allowed to delete a nonexistent scope.
            scope._remove_from_pool()

    def test_size(self):
        paddle_c = paddle.fluid.core
        scope = paddle_c.Scope()
        var_a = scope.var("var_a")
        self.assertEqual(scope.size(), 1)
        self.assertIsNotNone(scope.find_var('var_a'))


if __name__ == '__main__':
    unittest.main()
