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
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid.op import Operator


class TestSplitIdsOp(OpTest):
    def setUp(self):
        self.op_type = "split_ids"
        ids = np.array([[0], [2], [2], [3], [5], [5], [6]]).astype('int64')
        out0 = np.array([[0], [3], [6]]).astype('int64')
        out1 = np.array([[]]).astype('int64')
        out2 = np.array([[2], [2], [5], [5]]).astype('int64')
        self.inputs = {'Ids': ids}
        self.outputs = {'Out': [('out0', out0), ('out1', out1), ('out2', out2)]}

    def test_check_output(self):
        self.check_output()


class TestSpliteIds(unittest.TestCase):
    def get_places(self):
        places = [core.CPUPlace()]
        return places

    def test_check_output(self):
        for place in self.get_places():
            self.check_with_place(place)

    def check_with_place(self, place):
        scope = core.Scope()
        rows = [0, 5, 7, 4, 9]
        height = 20
        row_numel = 2

        # initialize input variable X
        x = scope.var('X').get_selected_rows()
        x.set_rows(rows)
        x.set_height(height)
        np_array = np.ones((len(rows), row_numel)).astype("float32")
        for i in range(len(rows)):
            for j in range(row_numel):
                np_array[i, j] = rows[i] + j
        x_tensor = x.get_tensor()
        x_tensor.set(np_array, place)

        outs_name = ["out%d" % i for i in xrange(3)]
        outs = [
            scope.var(var_name).get_selected_rows() for var_name in outs_name
        ]

        # expected output selected rows
        expected_out0_rows = [0, 9]
        expected_out1_rows = [7, 4]
        expected_out2_rows = [5]

        op = Operator("split_ids", Ids="X", Out=outs_name)

        op.run(scope, place)

        self.assertEqual(outs[0].rows(), expected_out0_rows)
        self.assertEqual(outs[1].rows(), expected_out1_rows)
        self.assertEqual(outs[2].rows(), expected_out2_rows)

        self.assertAlmostEqual(0.0, np.array(outs[0].get_tensor())[0, 0])
        self.assertAlmostEqual(1.0, np.array(outs[0].get_tensor())[0, 1])
        self.assertAlmostEqual(9.0, np.array(outs[0].get_tensor())[1, 0])
        self.assertAlmostEqual(10.0, np.array(outs[0].get_tensor())[1, 1])

        self.assertAlmostEqual(7.0, np.array(outs[1].get_tensor())[0, 0])
        self.assertAlmostEqual(8.0, np.array(outs[1].get_tensor())[0, 1])
        self.assertAlmostEqual(4.0, np.array(outs[1].get_tensor())[1, 0])
        self.assertAlmostEqual(5.0, np.array(outs[1].get_tensor())[1, 1])

        self.assertAlmostEqual(5.0, np.array(outs[2].get_tensor())[0, 0])
        self.assertAlmostEqual(6.0, np.array(outs[2].get_tensor())[0, 1])


if __name__ == '__main__':
    unittest.main()
