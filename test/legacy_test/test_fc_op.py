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
from op_test import OpTest, paddle_static_guard

import paddle
from paddle import base
from paddle.base import Program, core, program_guard

SEED = 2020


def round_array(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i,j] > 0:
                x[i,j] = np.math.ceil(x[i,j])
            else:
                x[i,j] = np.math.floor(x[i,j])

def round_array_with_ties_to_even(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x = x[i,j]
            xLower = np.math.floor(x)
            xUpper = np.math.ceil(x)
            dLower = x - xLower
            dUpper = xUpper - x
            if dLower == dUpper:
                if xLower % 2 == 0:
                    x[i,j] = xLower
                else:
                    x[i,j] = xUpper
            else:
                if dLower < dUpper:
                    x[i,j] = xLower
                else:
                    x[i,j] = xUpper

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
    
def fc_quant_refer(matrix, with_bias, scale_in, scale_weights, quant_round_type = 1, quant_max_bound = 127, quant_min_bound = -127, with_relu=False):
    in_n, in_c, in_h, in_w = matrix.input.shape
    w_i, w_o = matrix.weights.shape

    x_data = np.reshape(matrix.input, [in_n, in_c * in_h * in_w])
    quant_x_data = x_data.astype('float32')
    quant_x_data = quant_x_data * quant_max_bound * scale_in
    if quant_round_type == 0:
        round_array_with_ties_to_even(quant_x_data)
    else:
        round_array(quant_x_data)
    quant_x_data[quant_x_data > quant_max_bound] = quant_max_bound
    quant_x_data[quant_x_data < quant_min_bound] = quant_min_bound
    quant_x_data = quant_x_data.astype('int8')

    w_data = np.reshape(matrix.weights, [w_i, w_o])
    b_data = np.reshape(matrix.bias, [1, w_o])
    result = None
    quant_result = np.matmul(quant_x_data.astype('int32'), w_data.astype('int32'))
    scale_out = scale_weights
    for i in range(len(scale_out)):
        scale_out[i] *= scale_in
    result = quant_result / quant_max_bound / quant_max_bound / scale_out
    result = result.astype(x_data.dtype)

    if with_bias:
        result = result + b_data

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

def get_scale_in(input):
    max_v = np.max(np.abs(input))
    return 1 / max_v

def get_scale_weights(weights):
    max_v = np.max(np.abs(weights), axis=0)
    return 1 / max_v

def quant_weights(weights, scale_weights, quant_round_type, quant_max_bound, quant_min_bound):
    quant_weights = weights.astype('float32')
    quant_weights = (quant_max_bound * scale_weights * quant_weights)
    if quant_round_type == 0:
        round_array_with_ties_to_even(quant_weights)
    else:
        round_array(quant_weights)
    quant_weights[quant_weights > quant_max_bound] = quant_max_bound
    quant_weights[quant_weights < quant_min_bound] = quant_min_bound
    quant_weights = quant_weights.astype('int8')
    return quant_weights

class TestFCOp(OpTest):
    def config(self):
        self.with_bias = True
        self.with_relu = True
        self.matrix = MatrixGenerate(1, 10, 15, 3, 3, 2)

    def setUp(self):
        self.op_type = "fc"
        self.config()

        if self.with_bias:
            self.inputs = {
                'Input': self.matrix.input,
                'W': self.matrix.weights,
                'Bias': self.matrix.bias,
            }
        else:
            self.inputs = {'Input': self.matrix.input, 'W': self.matrix.weights}

        if self.with_relu:
            activation_type = "relu"
        else:
            activation_type = ""
        if hasattr(self, 'is_quant'):
            self.attrs = {'use_mkldnn': False, 'activation_type': activation_type, 'is_quant': self.is_quant, 'quant_round_type': self.quant_round_type, 'quant_max_bound': self.quant_max_bound, 'quant_min_bound': self.quant_min_bound, 'Scale_in' : self.scale_in, 'Scale_weights' : self.scale_weights}
        else:
            self.attrs = {'use_mkldnn': False, 'activation_type': activation_type}

        if hasattr(self, 'is_quant') and self.attrs['is_quant']:
            self.outputs = {
                'Out': fc_quant_refer(self.matrix, self.with_bias, self.scale_in, self.scale_weights, self.quant_round_type, self.quant_max_bound, self.quant_min_bound, self.with_relu)
            }
        else:
            self.outputs = {
                'Out': fc_refer(self.matrix, self.with_bias, self.with_relu)
            }

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_dygraph=False)


class TestFCOpQuantNoBias1(TestFCOp):
    def config(self):
        self.with_bias = False
        self.with_relu = False
        self.is_quant = True
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(16, 10, 16, 4, 4, 2)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(self.matrix.weights, self.scale_weights, self.quant_round_type, self.quant_max_bound, self.quant_min_bound)

class TestFCOpQuantBias2(TestFCOp):
    def config(self):
        self.with_bias = True
        self.with_relu = True
        self.is_quant = True
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(1, 64, 32, 3, 3, 1)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(self.matrix.weights, self.scale_weights, self.quant_round_type, self.quant_max_bound, self.quant_min_bound)

class TestFCOpQuantWithPadding(TestFCOp):
    def config(self):
        self.with_bias = True
        self.with_relu = True
        self.is_quant = True
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(1, 4, 4, 128, 128, 2)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(self.matrix.weights, self.scale_weights, self.quant_round_type, self.quant_max_bound, self.quant_min_bound)

class TestFCOpNoBias1(TestFCOp):
    def config(self):
        self.with_bias = False
        self.with_relu = False
        self.matrix = MatrixGenerate(2, 8, 10, 1, 1, 2)


class TestFCOpNoBias2(TestFCOp):
    def config(self):
        self.with_bias = False
        self.with_relu = False
        self.matrix = MatrixGenerate(4, 5, 6, 2, 2, 1)


class TestFCOpNoBias4(TestFCOp):
    def config(self):
        self.with_bias = False
        self.with_relu = False
        self.matrix = MatrixGenerate(1, 32, 64, 3, 3, 1)


class TestFCOpWithBias1(TestFCOp):
    def config(self):
        self.with_bias = True
        self.with_relu = False
        self.matrix = MatrixGenerate(3, 8, 10, 2, 1, 2)


class TestFCOpWithBias2(TestFCOp):
    def config(self):
        self.with_bias = True
        self.with_relu = True
        self.matrix = MatrixGenerate(4, 5, 6, 2, 2, 1)


class TestFCOpWithBias3(TestFCOp):
    def config(self):
        self.with_bias = True
        self.with_relu = True
        self.matrix = MatrixGenerate(1, 64, 32, 3, 3, 1)


class TestFCOpWithPadding(TestFCOp):
    def config(self):
        self.with_bias = True
        self.with_relu = True
        self.matrix = MatrixGenerate(1, 4, 3, 128, 128, 2)


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
