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
from op_test import OpTest


class TestExtractRows(OpTest):
    def check_with_place(self, place):
        scope = core.Scope()

        # create and initialize Variable
        feature_len = 12
        rows = [0, 4, 4, 7]
        np_array = np.ones((len(rows), feature_len)).astype("float32")

        in_x = scope.var('X').get_selected_rows()
        in_x.set_height(len(rows))
        in_x.set_rows(rows)
        in_x_tensor = in_x.get_tensor()
        in_x_tensor.set(np_array, place)

        # create Out Variable
        out_tensor = scope.var('Out').get_tensor()

        # create and run lookup_table operator
        extract_rows_op = Operator("extract_rows", X='X', Out='Out')
        extract_rows_op.run(scope, place)

        # get result from Out
        result_array = np.array(out_tensor)
        result_array = [ele[0] for ele in result_array]
        assert result_array == rows

    def test_concat_rows(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place)


if __name__ == '__main__':
    unittest.main()
