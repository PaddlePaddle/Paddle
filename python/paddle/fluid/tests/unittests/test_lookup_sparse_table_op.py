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


class TestLookupSpraseTable(OpTest):
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

        # create and initialize Id Variable
        ids = scope.var("Ids").get_tensor()
        ids_array1 = np.array([0, 2, 3, 2, 5, 0, 100]).astype("int64")
        ids.set(ids_array1, place)

        # create Out Variable
        out_tensor = scope.var('Out').get_tensor()

        # create and run lookup_table operator
        lookup_table = Operator(
            "lookup_sparse_table",
            W='W',
            Ids='Ids',
            Out='Out',
            min=-5.0,
            max=10.0,
            seed=10)
        lookup_table.run(scope, place)

        # get result from Out
        result_array1 = np.array(out_tensor)
        # all(): return True if all elements of the iterable are true (or if the iterable is empty)
        assert (result_array1[0] == w_array[0]).all()
        assert (result_array1[1] == w_array[1]).all()
        assert (result_array1[2] == w_array[2]).all()
        assert (result_array1[3] == w_array[1]).all()
        assert (result_array1[4] == w_array[3]).all()
        assert (result_array1[5] == w_array[0]).all()
        assert (result_array1[6] == w_array[4]).all()

        # create and initialize Id Variable
        ids = scope.var("Ids").get_tensor()
        ids_array2 = np.array([4, 2, 3, 7, 100000]).astype("int64")
        ids.set(ids_array2, place)
        lookup_table.run(scope, place)

        result_array2 = np.array(out_tensor)
        assert (result_array2[0] == w_array[5]).all()
        assert (result_array2[1] == w_array[1]).all()
        assert (result_array2[2] == w_array[2]).all()
        assert (result_array2[3] == w_array[6]).all()
        assert (result_array2[4] == w_array[7]).all()

        # create and run lookup_table operator
        test_lookup_table = Operator(
            "lookup_sparse_table",
            W='W',
            Ids='Ids',
            Out='Out',
            min=-5.0,
            max=10.0,
            seed=10,
            is_test=True)

        ids = scope.var("Ids").get_tensor()
        unknown_id = [44, 22, 33]
        ids_array2 = np.array([4, 2, 3, 7, 100000] + unknown_id).astype("int64")
        ids.set(ids_array2, place)
        test_lookup_table.run(scope, place)

        result_array2 = np.array(out_tensor)
        assert (result_array2[0] == w_array[5]).all()
        assert (result_array2[1] == w_array[1]).all()
        assert (result_array2[2] == w_array[2]).all()
        assert (result_array2[3] == w_array[6]).all()
        assert (result_array2[4] == w_array[7]).all()

        for i in [5, 6, 7]:
            assert np.all(result_array2[i] == 0)

    def test_w_is_selected_rows(self):
        places = [core.CPUPlace()]
        # currently only support CPU
        for place in places:
            self.check_with_place(place)


if __name__ == "__main__":
    unittest.main()
