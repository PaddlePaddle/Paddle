# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core


class TestSparseSquareOp(unittest.TestCase):
    def check_with_place(self, place):
        scope = core.Scope()

        # create and initialize Grad Variable
        height = 10
        rows = [0, 4, 7]
        self.row_numel = 12

        x_selected_rows = scope.var('X').get_selected_rows()
        x_selected_rows.set_height(height)
        x_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), self.row_numel)).astype("float32")
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0

        x_tensor = x_selected_rows.get_tensor()
        x_tensor.set(np_array, place)

        out_selected_rows = scope.var('Out').get_selected_rows()
        # create and run sqrt operator
        square_op = Operator("square", X='X', Out='Out')
        square_op.run(scope, place)

        # get and compare result
        result_array = np.array(out_selected_rows.get_tensor())

        np.testing.assert_array_equal(result_array, np.square(np_array))

    def test_sparse_acti(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(core.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place)


class TestSparseSqrtOp(unittest.TestCase):
    def check_with_place(self, place):
        scope = core.Scope()

        # create and initialize Grad Variable
        height = 10
        rows = [0, 4, 7]
        self.row_numel = 12

        x_selected_rows = scope.var('X1').get_selected_rows()
        x_selected_rows.set_height(height)
        x_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), self.row_numel)).astype("float32")
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0

        x_tensor = x_selected_rows.get_tensor()
        x_tensor.set(np_array, place)

        out_selected_rows = scope.var('Out1').get_selected_rows()
        # create and run sqrt operator
        sqrt_op = Operator("sqrt", X='X1', Out='Out1')
        sqrt_op.run(scope, place)

        # get and compare result
        result_array = np.array(out_selected_rows.get_tensor())
        np.testing.assert_allclose(result_array, np.sqrt(np_array), rtol=1e-05)

    def test_sparse_acti(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(core.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
