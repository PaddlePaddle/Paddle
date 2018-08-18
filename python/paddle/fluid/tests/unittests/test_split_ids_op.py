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
import six
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


class TestSplitSelectedRows(unittest.TestCase):
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

        outs_name = ["out%d" % i for i in six.moves.xrange(3)]
        outs = [
            scope.var(var_name).get_selected_rows() for var_name in outs_name
        ]

        # expected output selected rows
        expected_out_rows = [[0, 9], [7, 4], [5]]

        op = Operator("split_ids", Ids="X", Out=outs_name)

        for _ in range(3):
            op.run(scope, place)

            for i in range(len(outs)):
                expected_rows = expected_out_rows[i]
                self.assertEqual(outs[i].rows(), expected_rows)
                for j in range(len(expected_rows)):
                    row = expected_rows[j]
                    self.assertAlmostEqual(
                        float(row), np.array(outs[i].get_tensor())[j, 0])
                    self.assertAlmostEqual(
                        float(row + 1), np.array(outs[i].get_tensor())[j, 1])


class TestSpliteIdsByRefs(unittest.TestCase):
    def check_with_place(self, place):
        scope = core.Scope()

        # create and initialize W Variable
        table_size = 10000
        row_numel = 8

        w_selected_rows = scope.var('W').get_selected_rows()
        w_selected_rows.set_height(table_size)
        w_array = np.ones((table_size, row_numel)).astype("float32")
        for i in range(table_size):
            w_array[i] *= i
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)

        def create_lod_tensor(name, data, lod, scope, place):
            lod_tensor = scope.var(name).get_tensor()
            lod_tensor.set(data, place)
            if lod is not None:
                lod_tensor.set_lod(lod)
            return lod_tensor

        embedding_out_data = np.ones((10, 3)).astype("float32")
        for i in range(10):
            embedding_out_data[i] *= i
        embedding_out_t = create_lod_tensor("embedding_out", embedding_out_data,
                                            None, scope, place)
        id1_t = create_lod_tensor('id1',
                                  np.array([[1], [1], [1]]).astype('int64'),
                                  [[0, 1, 3]], scope, place)
        id2_t = create_lod_tensor('id2',
                                  np.array([[1], [1], [1]]).astype('int64'),
                                  [[0, 2, 3]], scope, place)
        id3_t = create_lod_tensor('id3',
                                  np.array(
                                      [[1], [1], [1], [1]]).astype('int64'),
                                  [[0, 2, 4]], scope, place)

        # create Out Variable
        out1_t = scope.var("out1").get_tensor()
        out2_t = scope.var("out2").get_tensor()
        out3_t = scope.var("out3").get_tensor()

        # create and run operator
        split_ids_op = Operator(
            "split_ids",
            Ids='embedding_out',
            Refs=['id1', 'id2', 'id3'],
            Out=['out1', 'out2', 'out3'])
        split_ids_op.run(scope, place)

        out_concated = np.concatenate(
            (np.array(out1_t), np.array(out2_t), np.array(out3_t)), axis=0)
        self.assertTrue((embedding_out_data == out_concated).all())
        self.assertEqual(id1_t.lod(), out1_t.lod())
        self.assertEqual(id2_t.lod(), out2_t.lod())
        self.assertEqual(id3_t.lod(), out3_t.lod())

    def test_split_lod_tensor_by_ref(self):
        places = [core.CPUPlace()]
        # currently only support CPU
        for place in places:
            self.check_with_place(place)


if __name__ == '__main__':
    unittest.main()
