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
import paddle.fluid.core as core
from paddle.fluid.op import Operator


class TestLookupSpraseTable(unittest.TestCase):
    def check_with_place(self, place):
        scope = core.Scope()

        rows = [0, 1, 2, 3, 4, 5, 6]
        row_numel = 7

        w_selected_rows = scope.var('W').get_selected_rows()
        w_selected_rows.set_height(len(rows))
        w_selected_rows.set_rows(rows)
        w_array = np.ones((len(rows), row_numel)).astype("float32")
        for i in range(len(rows)):
            w_array[i] *= i
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)

        # create and initialize Id Variable
        ids = scope.var("Ids").get_tensor()

        # create and run lookup_table operator
        lookup_table = Operator(
            "lookup_sparse_table_grad_split",
            Grad='W',
            Row={'Ids'},
            Value={'W'},
            is_entry=False,
            tablename="sparse")
        lookup_table.run(scope, place)

        # get result from Out
        result_array1 = np.array(ids)
        print(result_array1)
        print("== = = == == = == ==== ==== === ")
        value = scope.var("W").get_tensor()
        result_array1 = np.array(value)
        print(result_array1.shape)
        print(result_array1)

    def test_w_is_selected_rows(self):
        places = [core.CPUPlace()]
        # currently only support CPU
        for place in places:
            self.check_with_place(place)


if __name__ == "__main__":
    unittest.main()
