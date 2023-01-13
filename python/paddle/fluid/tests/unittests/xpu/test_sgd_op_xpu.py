#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

import numpy as np

sys.path.append("..")
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator


class XPUTestSgdOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'sgd'
        self.use_dynamic_create_class = False

    class TestSGDOp(XPUOpTest):
        def setUp(self):
            self.op_type = "sgd"
            self.dtype = self.in_type
            self.conf()
            w = np.random.random((self.h, self.w)).astype(self.dtype)
            g = np.random.random((self.h, self.w)).astype(self.dtype)
            lr = np.array([0.1]).astype(self.dtype)

            self.inputs = {'Param': w, 'Grad': g, 'LearningRate': lr}
            self.outputs = {'ParamOut': w - lr * g}

        def conf(self):
            self.h = 102
            self.w = 105

        def test_check_output_with_place(self):
            self.check_output_with_place(paddle.XPUPlace(0))

    class TestSGDOpCase8X(TestSGDOp):
        def conf(self):
            self.h = 10
            self.w = 64


support_types = get_xpu_op_support_types('sgd')
for stype in support_types:
    create_test_class(globals(), XPUTestSgdOp, stype)


class TestSGDOpWithLargeInput(unittest.TestCase):
    def runTest(self):
        data = fluid.layers.fill_constant(shape=[1], value=128, dtype='int64')
        label = fluid.layers.fill_constant(
            shape=[1, 150], value=0.5, dtype='float32'
        )
        emb = fluid.embedding(input=data, size=(10000, 150), dtype='float32')
        out = paddle.nn.functional.normalize(x=emb, axis=-1)

        cost = paddle.nn.functional.square_error_cost(input=out, label=label)
        avg_cost = paddle.mean(cost)
        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        sgd_optimizer.minimize(avg_cost)

        place = paddle.XPUPlace(0)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        result = exe.run(fluid.default_main_program(), fetch_list=[avg_cost])


class TestSparseSGDOp(unittest.TestCase):
    def check_with_place(self, place):
        scope = core.Scope()

        # create and initialize Grad Variable
        height = 10
        rows = [0, 4, 7]
        self.conf()

        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), self.row_numel)).astype("float32")
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(np_array, place)

        # create and initialize Param Variable
        param = scope.var('Param').get_tensor()
        param_array = np.full((height, self.row_numel), 5.0).astype("float32")
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
            LearningRate='LearningRate',
        )
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
        places = [core.XPUPlace(0)]
        for place in places:
            self.check_with_place(place)

    def conf(self):
        self.row_numel = 12


class TestSparseSGDOpCase8X(TestSparseSGDOp):
    def conf(self):
        self.row_numel = 16


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
