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
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from op_test import OpTest


class TestSGDOp(OpTest):
    def setUp(self):
        self.op_type = "sgd"
        w = np.random.random((102, 105)).astype("float32")
        g = np.random.random((102, 105)).astype("float32")
        lr = np.array([0.1]).astype("float32")

        self.inputs = {'Param': w, 'Grad': g, 'LearningRate': lr}
        self.outputs = {'ParamOut': w - lr * g}

    def test_check_output(self):
        self.check_output()


class TestSparseSGDOp(unittest.TestCase):
    def check_with_place(self, place):
        scope = core.Scope()

        # create and initialize Grad Variable   
        height = 10
        rows = [0, 4, 7]
        row_numel = 12

        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), row_numel)).astype("float32")
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(np_array, place)

        # create and initialize Param Variable
        param = scope.var('Param').get_tensor()
        param_array = np.full((height, row_numel), 5.0).astype("float32")
        param.set(param_array, place)

        # create and initialize LeraningRate Variable
        lr = scope.var('LearningRate').get_tensor()
        lr_array = np.full((1), 2.0).astype("float32")
        lr.set(lr_array, place)

        # create and run sgd operator
        sgd_op = Operator(
            "sgd",
            Param='Param',
            Grad='Grad',
            ParamOut='Param',
            LearningRate='LearningRate')
        sgd_op.run(scope, place)

        # get and compare result
        result_array = np.array(param)

        # rows[0] = 0, 5.0 - 2.0 * 2.0
        self.assertAlmostEqual(1.0, result_array[rows[0], 0])
        # rows[0] = 0, 5.0 - 2.0 * 1.0
        self.assertAlmostEqual(3.0, result_array[rows[0], 2])
        # 5.0 - 2.0 * 0.0
        self.assertAlmostEqual(5.0, result_array[1, 0])
        # rows[1] = 4, 5.0 - 2.0 * 1.0
        self.assertAlmostEqual(3.0, result_array[rows[1], 10])
        # 5.0 - 2.0 * 0.0
        self.assertAlmostEqual(5.0, result_array[5, 8])
        # rows[2] = 7, 5.0 - 2.0 * 1.0
        self.assertAlmostEqual(3.0, result_array[rows[2], 1])
        # rows[2] = 7, 5.0 - 2.0 * 4.0
        self.assertAlmostEqual(-3.0, result_array[rows[2], 8])

    def test_sparse_sgd(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place)


class TestSGDOpOptimizeSelectedRows(unittest.TestCase):
    def check_with_place(self, place):
        scope = core.Scope()

        row_width = 12
        # create and initialize Grad Variable
        grad_height = 10
        grad_rows = [0, 4, 7]

        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(grad_height)
        grad_selected_rows.set_rows(grad_rows)
        grad_array = np.ones((len(grad_rows), row_width)).astype("float32")
        grad_array[0, 0] = 2.0
        grad_array[2, 8] = 4.0

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(grad_array, place)

        # create and initialize Param Variable
        # create and initialize W Variable
        param_rows = [0, 1, 2, 3, 4, 5, 6, 7]

        # init Param
        w_selected_rows = scope.var('Param').get_selected_rows()
        w_selected_rows.set_height(len(param_rows))
        w_selected_rows.set_rows(param_rows)
        w_array = np.ones((len(param_rows), row_width)).astype("float32")
        for i in range(len(param_rows)):
            w_array[i] *= i
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)

        w_before_optimize = np.array(w_tensor)

        # create and initialize LeraningRate Variable
        lr_value = 0.1
        lr = scope.var('LearningRate').get_tensor()
        lr_array = np.full((1), lr_value).astype("float32")
        lr.set(lr_array, place)

        # optimize with Python
        w_after_optimize = np.copy(w_before_optimize)
        for index, id in enumerate(grad_rows):
            w_after_optimize[id] = w_before_optimize[
                id] - lr_value * grad_array[index]

        # create and run sgd operator
        sgd_op = Operator(
            "sgd",
            Param='Param',
            Grad='Grad',
            ParamOut='Param',
            LearningRate='LearningRate')
        sgd_op.run(scope, place)

        # get and compare result
        result_array = np.array(w_tensor)
        assert (result_array == w_after_optimize).all()

    def test_sparse_parameter_sgd(self):
        places = [core.CPUPlace()]
        # do not support GPU kernel currently
        for place in places:
            self.check_with_place(place)


if __name__ == "__main__":
    unittest.main()
