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


class TestSparseSparseParameterExtractOp(unittest.TestCase):
    def check_with_place(self, place):
        scope = core.Scope()

        height = 10
        rows = [0, 4, 7]
        row_numel = 12

        # create and initialize Param Variable
        param_tensor = scope.var('Param').get_tensor()
        param_array = np.random.random((height, row_numel)).astype("float32")
        param_tensor.set(param_array, place)

        # create and initialize Grad Variable
        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        np_array = np.random.random((len(rows), row_numel)).astype("float32")

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(np_array, place)

        param_out_selected_rows = scope.var('ParamOut').get_selected_rows()

        # create and run sparse_parameter_extract op
        sparse_parameter_extract_op = Operator(
            "sparse_parameter_extract",
            Param='Param',
            Grad='Grad',
            ParamOut='ParamOut')
        sparse_parameter_extract_op.run(scope, place)

        self.assertEqual(param_out_selected_rows.height(), height)
        self.assertEqual(param_out_selected_rows.rows(), rows)

        # get and compare result
        result_array = np.array(param_out_selected_rows.get_tensor())
        for i in range(len(rows)):
            self.assertTrue((result_array[i] == param_array[rows[i]]).all())

    def test_sparse_parameter_extract(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place)


if __name__ == "__main__":
    unittest.main()
