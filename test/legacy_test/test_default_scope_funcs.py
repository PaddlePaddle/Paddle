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

import unittest

from paddle.base.default_scope_funcs import (
    enter_local_scope,
    find_var,
    get_cur_scope,
    leave_local_scope,
    scoped_function,
    var,
)


class TestDefaultScopeFuncs(unittest.TestCase):
    def test_cur_scope(self):
        self.assertIsNotNone(get_cur_scope())

    def test_none_variable(self):
        self.assertIsNone(find_var("test"))

    def test_create_var_get_var(self):
        var_a = var("var_a")
        self.assertIsNotNone(var_a)
        self.assertIsNotNone(get_cur_scope().find_var('var_a'))
        enter_local_scope()
        self.assertIsNotNone(get_cur_scope().find_var('var_a'))
        leave_local_scope()

    def test_var_get_int(self):
        def __new_scope__():
            i = var("var_i")
            self.assertFalse(i.is_int())
            i.set_int(10)
            self.assertTrue(i.is_int())
            self.assertEqual(10, i.get_int())

        for _ in range(10):
            scoped_function(__new_scope__)


if __name__ == '__main__':
    unittest.main()
