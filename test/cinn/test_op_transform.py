#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

from test_utils import SingleOpTester

import paddle
from paddle import static
from paddle.cinn import framework

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def matmul_util(inputs_data, input_shape, trans_a, trans_b, alpha):
    main_program = static.Program()
    paddle.enable_static()
    with static.program_guard(main_program, static.Program()):
        [input_x, input_y] = inputs_data
        x = static.data(name='x', shape=input_shape[0], dtype='float32')
        y = static.data(name='y', shape=input_shape[1], dtype='float32')
        output = paddle.matmul(x, y, trans_a, trans_b)
        output = paddle.scale(output, scale=alpha)
        exe = static.Executor(paddle.CPUPlace())
        exe.run(static.default_startup_program())
        (res,) = exe.run(
            static.default_main_program(),
            feed={'x': input_x, 'y': input_y},
            fetch_list=[output],
        )
        return res


class OpTest_matmul_0(SingleOpTester):
    def init_testcase(self):
        self.input_shape = [[100, 32], [32, 100]]
        self.output_shape = [[100, 100], [100, 100]]
        self.trans_a = False
        self.trans_b = False
        self.alpha = 1.0
        self.attrs = framework.NodeAttr()
        self.attrs.set_attr("trans_a", self.trans_a)
        self.attrs.set_attr("trans_b", self.trans_b)
        self.attrs.set_attr("alpha", self.alpha)

    def create_target_data(self, inputs_data, attrs):
        return matmul_util(
            inputs_data,
            self.input_shape,
            self.trans_a,
            self.trans_b,
            self.alpha,
        )

    def test_op(self):
        self.init_testcase()
        self.to_test_op(
            self.input_shape, self.output_shape, "matmul", self.attrs, 0
        )


class OpTest_matmul_1(SingleOpTester):
    def init_testcase(self):
        self.input_shape = [[100, 32], [100, 32]]
        self.output_shape = [[100, 100], [2, 32, 50]]
        self.trans_a = False
        self.trans_b = True
        self.alpha = 2.0
        self.attrs = framework.NodeAttr()
        self.attrs.set_attr("trans_a", self.trans_a)
        self.attrs.set_attr("trans_b", self.trans_b)
        self.attrs.set_attr("alpha", self.alpha)

    def create_target_data(self, inputs_data, attrs):
        return matmul_util(
            inputs_data,
            self.input_shape,
            self.trans_a,
            self.trans_b,
            self.alpha,
        )

    def test_op(self):
        self.init_testcase()
        self.to_test_op(
            self.input_shape, self.output_shape, "matmul", self.attrs, 0
        )


class OpTest_matmul_2(SingleOpTester):
    def init_testcase(self):
        self.input_shape = [[2, 3, 100, 32], [2, 3, 100, 32]]
        self.output_shape = [[2, 3, 100, 100], [2, 3, 2, 100, 16]]
        self.trans_a = False
        self.trans_b = True
        self.alpha = 2.0
        self.attrs = framework.NodeAttr()
        self.attrs.set_attr("trans_a", self.trans_a)
        self.attrs.set_attr("trans_b", self.trans_b)
        self.attrs.set_attr("alpha", self.alpha)

    def create_target_data(self, inputs_data, attrs):
        return matmul_util(
            inputs_data,
            self.input_shape,
            self.trans_a,
            self.trans_b,
            self.alpha,
        )

    def test_op(self):
        self.init_testcase()
        self.to_test_op(
            self.input_shape, self.output_shape, "matmul", self.attrs, 0
        )


class OpTest_matmul_3(SingleOpTester):
    def init_testcase(self):
        self.input_shape = [[32, 100], [32, 100]]
        self.output_shape = [[100, 100], [2, 100, 16]]
        self.trans_a = True
        self.trans_b = False
        self.alpha = 2.0
        self.attrs = framework.NodeAttr()
        self.attrs.set_attr("trans_a", self.trans_a)
        self.attrs.set_attr("trans_b", self.trans_b)
        self.attrs.set_attr("alpha", self.alpha)

    def create_target_data(self, inputs_data, attrs):
        return matmul_util(
            inputs_data,
            self.input_shape,
            self.trans_a,
            self.trans_b,
            self.alpha,
        )

    def test_op(self):
        self.init_testcase()
        self.to_test_op(
            self.input_shape, self.output_shape, "matmul", self.attrs, 0
        )


class OpTest_matmul_4(SingleOpTester):
    def init_testcase(self):
        self.input_shape = [[32, 100], [100]]
        self.output_shape = [[32], [2, 100, 16]]
        self.trans_a = False
        self.trans_b = False
        self.alpha = 2.0
        self.attrs = framework.NodeAttr()
        self.attrs.set_attr("trans_a", self.trans_a)
        self.attrs.set_attr("trans_b", self.trans_b)
        self.attrs.set_attr("alpha", self.alpha)

    def create_target_data(self, inputs_data, attrs):
        return matmul_util(
            inputs_data,
            self.input_shape,
            self.trans_a,
            self.trans_b,
            self.alpha,
        )

    def test_op(self):
        self.init_testcase()
        self.to_test_op(
            self.input_shape, self.output_shape, "matmul", self.attrs, 0
        )


class OpTest_matmul_5(SingleOpTester):
    def init_testcase(self):
        self.input_shape = [[100], [100]]
        self.output_shape = [[1], [1, 100, 1]]
        self.trans_a = False
        self.trans_b = False
        self.alpha = 2.0
        self.attrs = framework.NodeAttr()
        self.attrs.set_attr("trans_a", self.trans_a)
        self.attrs.set_attr("trans_b", self.trans_b)
        self.attrs.set_attr("alpha", self.alpha)

    def create_target_data(self, inputs_data, attrs):
        return matmul_util(
            inputs_data,
            self.input_shape,
            self.trans_a,
            self.trans_b,
            self.alpha,
        )

    def test_op(self):
        self.init_testcase()
        self.to_test_op(
            self.input_shape, self.output_shape, "matmul", self.attrs, 0
        )


class OpTest_matmul_6(SingleOpTester):
    def init_testcase(self):
        self.input_shape = [[32, 1], [1, 100]]
        self.output_shape = [[32, 100], [2, 1, 50]]
        self.trans_a = False
        self.trans_b = False
        self.alpha = 2.0
        self.attrs = framework.NodeAttr()
        self.attrs.set_attr("trans_a", self.trans_a)
        self.attrs.set_attr("trans_b", self.trans_b)
        self.attrs.set_attr("alpha", self.alpha)

    def create_target_data(self, inputs_data, attrs):
        return matmul_util(
            inputs_data,
            self.input_shape,
            self.trans_a,
            self.trans_b,
            self.alpha,
        )

    def test_op(self):
        self.init_testcase()
        self.to_test_op(
            self.input_shape, self.output_shape, "matmul", self.attrs, 0
        )


if __name__ == "__main__":
    unittest.main()
