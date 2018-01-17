#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
import unittest
import paddle.v2.fluid.core as core
import numpy as np
from paddle.v2.fluid.op import Operator


class TestSpliteAndMergeSelectedRows(unittest.TestCase):
    def test_check_output(self):
        places = [core.CPUPlace()]
        if core.is_compile_gpu():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place)

    def check_with_place(self, place):
        scope = core.Scope()
        rows = [0, 5, 7, 4]
        height = 10
        row_numel = 2

        # initialize input variable X
        x = scope.var('X').get_selected_rows()
        x.set_rows(rows)
        x.set_height(height)
        np_array = np.ones((len(rows), row_numel)).astype("float32")
        np_array[0, 0] = 2.0
        np_array[2, 1] = 4.0
        x_tensor = x.get_tensor()
        x_tensor.set(np_array, place)

        rows_section = [2, 2]
        height_section = []

        # initialize output variables [out0, out1]
        out0 = scope.var('out0').get_selected_rows()
        out1 = scope.var('out1').get_selected_rows()

        # expected output selected rows
        expected_out0_rows = [0, 5]
        expected_out1_rows = [7, 4]
        expected_height = height

        split_selected_rows = Operator(
            "split_selected_rows",
            X="X",
            Out=["out0", "out1"],
            rows_section=rows_section,
            height_section=height_section)

        split_selected_rows.run(scope, place)

        self.assertEqual(out0.rows(), expected_out0_rows)
        self.assertEqual(out1.rows(), expected_out1_rows)

        self.assertEqual(out0.height(), expected_height)
        self.assertEqual(out1.height(), expected_height)

        self.assertAlmostEqual(2.0, np.array(out0.get_tensor())[0, 0])
        self.assertAlmostEqual(4.0, np.array(out1.get_tensor())[0, 1])


if __name__ == "__main__":
    unittest.main()
