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

import paddle.fluid.core
from op_test import OpTest
import unittest
import six



class TestScope(OpTest):
    def setUp(self):
        self.op_type = "copy_cross_scope"
        self.attrs = {}
        # self.__class__.use_npu = True
        self.place = paddle.CUDAPlace(0)
        scope = self.test_copy_scope()
        self.inputs = {'X': scope}
        self.outputs = {'Out': scope}

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    def test_copy_scope(self):
        paddle_c = paddle.fluid.core
        scope = paddle_c.Scope()
        # var_a = scope.var("var_a")
        # self.assertIsNotNone(var_a)
        # self.assertIsNotNone(scope.find_var('var_a'))
        for num in range(3):
            tmp_scope = scope.new_scope()
            tmp_var = tmp_scope.var(f"var_{num}")
            tmp_var.set_int(int(num))
        return scope

if __name__ == '__main__':
    unittest.main()
