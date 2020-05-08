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

        # create and initialize Id Variable
        ids = scope.var("Ids").get_tensor()
        ids_array1 = np.array([0, 2, 3, 2, 5, 0, 100]).astype("int64")
        ids.set(ids_array1, place)

        # create Out Variable
        out_tensor = scope.var('Param').get_tensor()
        m1 = scope.var('Moment1').get_tensor()

        # create and run lookup_table operator
        lookup_table = Operator(
            "lookup_sparse_table",
            Ids='Ids',
            Out0='Param',
            Out1='Moment1',
            tablename="embedding")
        lookup_table.run(scope, place)

        # get result from Out
        result_array1 = np.array(out_tensor)
        print(result_array1)
        print("== = = == == = == ==== ==== === ")
        result_array1 = np.array(m1)
        print(result_array1)

    def test_w_is_selected_rows(self):
        places = [core.CPUPlace()]
        # currently only support CPU
        for place in places:
            self.check_with_place(place)


if __name__ == "__main__":
    unittest.main()
