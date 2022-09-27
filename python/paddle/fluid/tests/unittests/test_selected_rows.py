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

import paddle.fluid.core as core
import unittest
import numpy as np


class TestSelectedRows(unittest.TestCase):

    def test_selected_rows(self):
        place = core.CPUPlace()
        height = 10
        rows = [0, 4, 7]
        row_numel = 12
        selected_rows = core.SelectedRows(rows, height)
        np_array = np.ones((len(rows), row_numel)).astype("float32")
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0
        tensor = selected_rows.get_tensor()
        tensor.set(np_array, place)

        # compare rows
        self.assertEqual(0, selected_rows.rows()[0])
        self.assertEqual(4, selected_rows.rows()[1])
        self.assertEqual(7, selected_rows.rows()[2])

        # compare height
        self.assertEqual(10, selected_rows.height())

        # compare tensor
        self.assertAlmostEqual(2.0,
                               selected_rows.get_tensor()._get_float_element(0))
        self.assertAlmostEqual(1.0,
                               selected_rows.get_tensor()._get_float_element(1))
        self.assertAlmostEqual(
            4.0,
            selected_rows.get_tensor()._get_float_element(2 * row_numel + 8))


if __name__ == "__main__":
    unittest.main()
