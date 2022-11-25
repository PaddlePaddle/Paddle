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
import paddle.fluid.core as core
import numpy as np
from paddle.fluid.op import Operator


class TestMergeSelectedRows(unittest.TestCase):

    def get_places(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def check_with_place(self, place):
        scope = core.Scope()
        x_rows = [0, 5, 5, 4, 19]
        out_rows = [0, 4, 5, 19]
        height = 20
        row_numel = 2

        np_array = np.ones((len(x_rows), row_numel)).astype("float32")
        np_array[1, :] = 2.0
        np_array[2, :] = 3.0
        np_array[3, :] = 4.0

        # initialize input variable X
        x = scope.var('X').get_selected_rows()
        x.set_rows(x_rows)
        x.set_height(height)
        x_tensor = x.get_tensor()
        x_tensor.set(np_array, place)

        # initialize input variable Out
        out = scope.var("Out").get_selected_rows()

        op = Operator("merge_selected_rows", X="X", Out="Out")

        op.run(scope, place)

        self.assertEqual(out.rows(), out_rows)
        self.assertEqual(out.height(), height)

        out_array = np.array(out.get_tensor())
        self.assertEqual((4, 2), out_array.shape)

        assert (out_array[0, :] == 1.0).all()
        assert (out_array[1, :] == 4.0).all()
        assert (out_array[2, :] == 5.0).all()
        assert (out_array[3, :] == 1.0).all()

    def test_check_output(self):
        for place in self.get_places():
            self.check_with_place(place)


if __name__ == "__main__":
    unittest.main()
