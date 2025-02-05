# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import paddle_static_guard

import paddle
from paddle import base
from paddle.base import Program, core, program_guard

SEED = 2020


def fc_refer(matrix, with_bias, with_relu=False):
    in_n, in_c, in_h, in_w = matrix.input.shape
    w_i, w_o = matrix.weights.shape

    x_data = np.reshape(matrix.input, [in_n, in_c * in_h * in_w])
    w_data = np.reshape(matrix.weights, [w_i, w_o])
    b_data = np.reshape(matrix.bias, [1, w_o])
    result = None

    if with_bias:
        result = np.dot(x_data, w_data) + b_data
    else:
        result = np.dot(x_data, w_data)

    if with_relu:
        return np.maximum(result, 0)
    else:
        return result


class MatrixGenerate:
    def __init__(self, mb, ic, oc, h, w, bias_dims=2):
        self.input = np.random.random((mb, ic, h, w)).astype("float32")
        self.weights = np.random.random((ic * h * w, oc)).astype("float32")
        if bias_dims == 2:
            self.bias = np.random.random((1, oc)).astype("float32")
        else:
            self.bias = np.random.random(oc).astype("float32")


class TestFcOp_NumFlattenDims_NegOne(unittest.TestCase):
    def test_api(self):
        def run_program(num_flatten_dims):
            paddle.seed(SEED)
            np.random.seed(SEED)
            startup_program = Program()
            main_program = Program()

            with paddle_static_guard():
                with program_guard(main_program, startup_program):
                    input = np.random.random([2, 2, 25]).astype("float32")
                    x = paddle.static.data(
                        name="x",
                        shape=[2, 2, 25],
                        dtype="float32",
                    )

                    out = paddle.static.nn.fc(
                        x=x, size=1, num_flatten_dims=num_flatten_dims
                    )

                place = (
                    base.CPUPlace()
                    if not core.is_compiled_with_cuda()
                    else base.CUDAPlace(0)
                )
                exe = base.Executor(place=place)
                exe.run(startup_program)
                out = exe.run(main_program, feed={"x": input}, fetch_list=[out])
                return out

        res_1 = run_program(-1)
        res_2 = run_program(2)
        np.testing.assert_array_equal(res_1, res_2)


class TestFCOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            input_data = np.random.random((2, 4)).astype("float32")

            def test_Variable():
                with paddle_static_guard():
                    # the input type must be Variable
                    paddle.static.nn.fc(x=input_data, size=1)

            self.assertRaises(TypeError, test_Variable)

            def test_input_list():
                with paddle_static_guard():
                    # each of input(list) must be Variable
                    paddle.static.nn.fc(x=[input_data], size=1)

            self.assertRaises(TypeError, test_input_list)

            def test_type():
                with paddle_static_guard():
                    # dtype must be float32 or float64
                    x2 = paddle.static.data(
                        name='x2', shape=[-1, 4], dtype='int32'
                    )
                    paddle.static.nn.fc(x=x2, size=1)

            self.assertRaises(TypeError, test_type)

            with paddle_static_guard():
                # The input dtype of fc can be float16 in GPU, test for warning
                x3 = paddle.static.data(
                    name='x3', shape=[-1, 4], dtype='float16'
                )
                paddle.static.nn.fc(x=x3, size=1)


if __name__ == "__main__":
    unittest.main()
