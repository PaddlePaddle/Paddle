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

import os
import unittest

import numpy as np
from op import Operator

import paddle
from paddle.base import Program, core, program_guard
from paddle.nn import clip


class TestGetTensorFromSelectedRowsError(unittest.TestCase):
    """get_tensor_from_selected_rows error message enhance"""

    def test_errors(self):
        with program_guard(Program()):
            x_var = paddle.static.data('X', [2, 3])
            x_data = np.random.random((2, 4)).astype("float32")

            def test_Variable():
                clip.get_tensor_from_selected_rows(x=x_data)

            self.assertRaises(TypeError, test_Variable)

            def test_SELECTED_ROWS():
                clip.get_tensor_from_selected_rows(x=x_var)

            self.assertRaises(
                (TypeError, NotImplementedError), test_SELECTED_ROWS
            )


class TestGetTensorFromSelectedRows(unittest.TestCase):
    def get_places(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(core.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def check_with_place(self, place):
        scope = core.Scope()
        x_rows = [0, 5, 5, 4, 19]
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
        out = scope.var("Out").get_tensor()

        op = Operator("get_tensor_from_selected_rows", X="X", Out="Out")

        op.run(scope, place)

        out_array = np.array(out)
        self.assertEqual((5, 2), out_array.shape)
        assert (out_array == np_array).all()

    def test_check_output(self):
        for place in self.get_places():
            self.check_with_place(place)


if __name__ == "__main__":
    unittest.main()
