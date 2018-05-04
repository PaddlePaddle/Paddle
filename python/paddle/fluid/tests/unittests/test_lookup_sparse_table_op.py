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


def output_hist(out):
    hist, _ = np.histogram(out, range=(-5, 10))
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.1 * np.ones((10))
    return hist, prob


class TestLookupSpraseTable(OpTest):
    def check_with_place(self, place):
        scope = core.Scope()

        # create and initialize Id Variable
        ids = scope.var("Ids").get_tensor()
        ids_array = np.array([0, 2, 3, 5, 100]).astype("int64")
        ids.set(ids_array, place)

        # create and initialize W Variable
        rows = [0, 1, 2, 3, 4, 5, 6]
        row_numel = 10000

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
        result_array = np.array(out_tensor)
        # all(): return True if all elements of the iterable are true (or if the iterable is empty)
        for idx, row in enumerate(ids_array[:-2]):
            assert (row == result_array[idx]).all()

        # check the random value
        hist, prob = output_hist(result_array[-1])
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))

    def test_w_is_selected_rows(self):
        places = [core.CPUPlace()]
        # currently only support CPU
        for place in places:
            self.check_with_place(place)


if __name__ == "__main__":
    unittest.main()
