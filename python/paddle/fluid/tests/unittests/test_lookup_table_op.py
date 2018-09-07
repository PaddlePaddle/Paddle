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


class TestLookupTableOp(OpTest):
    def setUp(self):
        self.op_type = "lookup_table"
        table = np.random.random((17, 31)).astype("float32")
        ids = np.random.randint(0, 17, 4).astype("int64")
        ids_expand = np.expand_dims(ids, axis=1)
        self.inputs = {'W': table, 'Ids': ids_expand}
        self.outputs = {'Out': table[ids]}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['W'], 'Out', no_grad_set=set('Ids'))


class TestLookupTableOpWithPadding(TestLookupTableOp):
    def test_check_output(self):
        ids = np.squeeze(self.inputs['Ids'])
        padding_idx = np.random.choice(ids, 1)[0]
        self.outputs['Out'][ids == padding_idx] = np.zeros(31)
        self.attrs = {'padding_idx': long(padding_idx)}
        self.check_output()

    def test_check_grad(self):
        # Since paddings are not trainable and fixed in forward, the gradient of 
        # paddings makes no sense and we don't test the gradient here.
        pass


class TestLookupTableIdsIsSelectedRows(OpTest):
    def check_with_place(self, place):
        scope = core.Scope()

        # create and initialize Variable
        height = 10
        rows = [0, 4, 4, 7]
        row_numel = 12

        # create and initialize W Variable
        W = scope.var('W').get_tensor()
        W_array = np.full((height, row_numel), 1.0).astype("float32")
        for i in range(height):
            W_array[i] *= i
        W.set(W_array, place)

        # create and initialize Ids Variable
        ids_selected_rows = scope.var('Ids').get_selected_rows()
        ids_selected_rows.set_height(len(rows))
        ids_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), row_numel)).astype("float32")
        ids_tensor = ids_selected_rows.get_tensor()
        ids_tensor.set(np_array, place)

        # create Out Variable
        Out = scope.var('Out').get_selected_rows()

        # create and run lookup_table operator
        concat_rows_op = Operator("lookup_table", W='W', Ids='Ids', Out='Out')
        concat_rows_op.run(scope, place)

        # get result from Out
        Out_tensor = Out.get_tensor()
        result_array = np.array(Out_tensor)

        # all(): return True if all elements of the iterable are true (or if the iterable is empty)
        for idx, row in enumerate(rows):
            assert (row == result_array[idx]).all()

    def test_concat_rows(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place)


class TestLookupTableWIsSelectedRows(OpTest):
    def check_with_place(self, place):
        scope = core.Scope()

        # create and initialize Id Variable
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.array([[0], [4], [3], [5]]).astype("int64")
        ids_tensor.set(ids_array, place)

        # create and initialize W Variable
        rows = [0, 1, 2, 3, 4, 5, 6]
        row_numel = 12

        w_selected_rows = scope.var('W').get_selected_rows()
        w_selected_rows.set_height(len(rows))
        w_selected_rows.set_rows(rows)
        w_array = np.ones((len(rows), row_numel)).astype("float32")
        for i in range(len(rows)):
            w_array[i] *= i
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)

        # create Out Variable
        out_tensor = scope.var('Out').get_tensor()

        # create and run lookup_table operator
        lookup_table = Operator("lookup_table", W='W', Ids='Ids', Out='Out')
        lookup_table.run(scope, place)

        # get result from Out
        result_array = np.array(out_tensor)
        # all(): return True if all elements of the iterable are true (or if the iterable is empty)
        for idx, row in enumerate(ids_array):
            assert (row[0] == result_array[idx]).all()

    def test_w_is_selected_rows(self):
        places = [core.CPUPlace()]
        # currently only support CPU
        for place in places:
            self.check_with_place(place)


if __name__ == "__main__":
    unittest.main()
