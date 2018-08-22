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
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid.op import Operator


class TestSumOp(OpTest):
    def setUp(self):
        self.op_type = "sum"
        self.use_mkldnn = False
        self.init_kernel_type()
        x0 = np.random.random((3, 4)).astype('float32')
        x1 = np.random.random((3, 4)).astype('float32')
        x2 = np.random.random((3, 4)).astype('float32')
        self.inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
        y = x0 + x1 + x2
        self.outputs = {'Out': y}
        self.attrs = {'use_mkldnn': self.use_mkldnn}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out')

    def init_kernel_type(self):
        pass


class TestSelectedRowsSumOp(OpTest):
    def check_with_place(self, place):
        scope = core.Scope()
        self.check_input_and_optput(scope, place, True, True, True)
        self.check_input_and_optput(scope, place, False, True, True)
        self.check_input_and_optput(scope, place, False, False, True)
        self.check_input_and_optput(scope, place, False, False, False)

    def check_input_and_optput(self,
                               scope,
                               place,
                               w1_has_data=False,
                               w2_has_data=False,
                               w3_has_data=False):

        self.create_selected_rows(scope, place, "W1", w1_has_data)
        self.create_selected_rows(scope, place, "W2", w2_has_data)
        self.create_selected_rows(scope, place, "W3", w3_has_data)

        # create Out Variable
        out = scope.var('Out').get_selected_rows()

        # create and run sum operator
        sum_op = Operator("sum", X=["W1", "W2", "W3"], Out='Out')
        sum_op.run(scope, place)

        has_data_w_num = 0
        for w in [w1_has_data, w2_has_data, w3_has_data]:
            if not w:
                has_data_w_num += 1

        self.assertEqual(7 * has_data_w_num, len(out.rows()))

    def create_selected_rows(self, scope, place, var_name, isEmpty):
        # create and initialize W Variable
        if not isEmpty:
            rows = [0, 1, 2, 3, 4, 5, 6]
            row_numel = 12
        else:
            rows = []
            row_numel = 12

        var = scope.var(var_name)
        w_selected_rows = var.get_selected_rows()
        w_selected_rows.set_height(len(rows))
        w_selected_rows.set_rows(rows)
        w_array = np.ones((len(rows), row_numel)).astype("float32")
        for i in range(len(rows)):
            w_array[i] *= i
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)

        return var

    def test_w_is_selected_rows(self):
        places = [core.CPUPlace()]
        # currently only support CPU
        for place in places:
            self.check_with_place(place)


if __name__ == "__main__":
    unittest.main()
