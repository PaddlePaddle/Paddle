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
from paddle.base.data_feeder import check_dtype
from paddle.base.framework import Variable, static_only
from paddle.common_ops_import import LayerHelper, check_type

SEED = 2020


@static_only
def quant_linear(
    x,
    w,
    size,
    scale_in,
    scale_weight,
    num_flatten_dims=1,
    bias_attr=None,
    activation=None,
    quant_round_type=1,
    quant_max_bound=127.0,
    quant_min_bound=-127.0,
    name=None,
):
    r"""

    Quant linear layer can take a tensor as its input and a tensor as the weight tensor.
    The quant linear layer multiplies the input tensor with the weight to produce
    an output tensor with shape :math:`[batch\_size, *, size]` , where :math:`*`
    means any number of additional dimensions. If :attr:`bias_attr` is not False, a 1-D bias tensor will
    be created and added to the output. If :attr:`activation` is not None,
    it will be applied to the output as well. Besides, the input tensor will be quantize to
    the tensor with int8 type, the parameter w must be a tensor with int8 type and the computation will also
    be with the int8 type.

    For a single input tensor :math:`X` , the equation is:

    .. math::

        Out = Act({XW + b})

    where:

    * :math:`X`: The input tensor.
    * :math:`W`: The weight matrix.
    * :math:`b`: The bias created by this layer (if needed).
    * :math:`Act`: The activation function.
    * :math:`Out`: The output tensor.

    Args:
        x (Tensor): A tensor. The number of dimensions
            of the tensor is at least 2. The data type should be float16, bfloat16, float32 or float64.
        w (Tensor): A tensor. The data type should be int8.
        size (int): The number of the output unit in this layer, which also means the feature
            size of output tensor.
        scale_in (float): The quantization scale for input.
        scale_weight (list[float]): The quantization scale for weights.
        num_flatten_dims (int, optional): The quant linear layer can accept an input tensor with more than
            two dimensions. If this happens, the multi-dimensional tensor will first be flattened
            into a 2-D matrix. The parameter :attr:`num_flatten_dims` determines how the input
            tensor is flattened: the first :math:`num\_flatten\_dims` (inclusive, index starts from 1)
            dimensions will be flatten to form the first dimension of the final matrix (height of
            the matrix), and the rest :math:`rank(x) - num\_flatten\_dims` dimensions are
            flattened to form the second dimension of the final matrix (width of the matrix).
            For example, assuming that :attr:`x` is a 5-dimensional tensor with a shape
            :math:`[2, 3, 4, 5, 6]` , and :attr:`num_flatten_dims` = 3.
            Then, the flattened matrix will have a shape :math:`[2 * 3 * 4, 5 * 6] = [24, 30]` .
            Default: 1.
        bias_attr (ParamAttr|bool, optional): The attribute of the learnable bias.
            If it is set to False, no bias will be added to the output.
            If it is set to None or one kind of ParamAttr, a bias parameter will
            be created according to ParamAttr. For detailed information, please refer
            to :attr:`paddle.ParamAttr`. The default value is None and the bias will be
            initialized to zero.
        activation (str, optional): Activation to be applied to the output of
            this layer. Only "relu" is supported. For more information,
            please refer to :ref:`api_guide_activations_en` . Default: None.
        quant_round_type (int, optional): The round type of float to int. 0 means rounding to nearest ties to even and 1 means rounding to nearest ties away from zero. Default: 1.
        quant_max_bound (float, optional): The max bound of float type to int type. Default: 127.0.
        quant_min_bound (float, optional): The min bound of float type to int type. Default: -127.0.
        name (str, optional): The default value is None. Normally there is no need for user to set
            it. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor, its shape is :math:`[batch\_size, *, size]` , and the data type is same with input.

    """

    def quant_linear_base(
        input,
        weight,
        size,
        scale_in,
        scale_weight,
        num_flatten_dims=1,
        bias_attr=None,
        act=None,
        quant_round_type=1,
        quant_max_bound=127.0,
        quant_min_bound=-127.0,
        name=None,
    ):
        helper = LayerHelper("quant_linear", **locals())
        check_type(input, 'input', Variable, 'quant_linear')
        dtype = helper.input_dtype()
        check_dtype(
            dtype,
            'input',
            ['float16', 'float32', 'float64'],
            'quant_linear',
        )

        input_shape = input.shape
        if num_flatten_dims == -1:
            num_flatten_dims = len(input_shape) - 1

        check_type(weight, "weight", Variable, 'quant_linear')
        check_dtype(
            weight.dtype,
            'weight',
            ['int8'],
            'quant_linear',
        )
        check_type(scale_weight, "scale_weight", list, 'quant_linear')
        if len(scale_weight) != size:
            raise AttributeError(
                "The length of scale_weight must be the same with the param size."
            )

        inputs_of_quant_linear = {"x": input, "w": weight}
        if bias_attr is not False:
            bias_shape = [size]
            bias = helper.create_parameter(
                attr=bias_attr, shape=bias_shape, dtype=dtype, is_bias=True
            )
            inputs_of_quant_linear["bias"] = bias

        out = helper.create_variable_for_type_inference(dtype)
        attrs_of_quant_linear = {
            "in_num_col_dims": num_flatten_dims,
            "activation_type": act,
            "scale_in": scale_in,
            "scale_weights": scale_weight,
            "quant_round_type": quant_round_type,
            "quant_max_bound": quant_max_bound,
            "quant_min_bound": quant_min_bound,
        }

        helper.append_op(
            type="quant_linear",
            inputs=inputs_of_quant_linear,
            outputs={"out": out},
            attrs=attrs_of_quant_linear,
        )
        return out

    return quant_linear_base(
        input=x,
        weight=w,
        size=size,
        scale_in=scale_in,
        scale_weight=scale_weight,
        num_flatten_dims=num_flatten_dims,
        bias_attr=bias_attr,
        act=activation,
        quant_round_type=quant_round_type,
        quant_max_bound=quant_max_bound,
        quant_min_bound=quant_min_bound,
        name=name,
    )


def round_array(x):
    x[x > 0] = np.ceil(x[x > 0])
    x[x <= 0] = np.floor(x[x <= 0])


def round_array_with_ties_to_even(x):
    xLower = np.floor(x)
    xUpper = np.ceil(x)
    dLower = x - xLower
    dUpper = xUpper - x
    x[(dLower == dUpper) & (xLower % 2 == 0)] = xLower[
        (dLower == dUpper) & (xLower % 2 == 0)
    ]
    x[(dLower == dUpper) & (xLower % 2 != 0)] = xUpper[
        (dLower == dUpper) & (xLower % 2 != 0)
    ]
    x[dLower < dUpper] = xLower[dLower < dUpper]
    x[dLower > dUpper] = xUpper[dLower > dUpper]


def quant_linear_refer(
    matrix,
    with_bias,
    scale_in,
    scale_weights,
    quant_round_type=1,
    quant_max_bound=127,
    quant_min_bound=-127,
    with_relu=False,
):
    in_n, in_c, in_h, in_w = matrix.input.shape
    w_i, w_o = matrix.weights.shape

    x_data = np.reshape(matrix.input, [in_n, in_c * in_h * in_w])
    quant_x_data = x_data.astype('float32')
    quant_x_data = quant_max_bound * scale_in * quant_x_data
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
    quant_result = np.dot(quant_x_data.astype('int32'), w_data.astype('int32'))
    scale_out = scale_weights * scale_in
    result = quant_result / (quant_max_bound * quant_max_bound * scale_out)
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


def quant_weights(
    weights, scale_weights, quant_round_type, quant_max_bound, quant_min_bound
):
    quant_weights = weights.astype('float32')
    quant_weights = quant_max_bound * scale_weights * quant_weights
    if quant_round_type == 0:
        round_array_with_ties_to_even(quant_weights)
    else:
        round_array(quant_weights)
    quant_weights[quant_weights > quant_max_bound] = quant_max_bound
    quant_weights[quant_weights < quant_min_bound] = quant_min_bound
    quant_weights = quant_weights.astype('int8')
    return quant_weights


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm(),
    "QuantLinear only supports cuda kernel.",
)
class TestQuantLinearOp(OpTest):
    def config(self):
        self.with_bias = False
        self.with_relu = False
        self.quant_round_type = 0
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(2, 1, 10, 1, 1, 2)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(
            self.matrix.weights,
            self.scale_weights,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )

    def setUp(self):
        self.op_type = "quant_linear"
        self.config()

        if self.with_bias:
            self.inputs = {
                'x': self.matrix.input,
                'w': self.matrix.weights,
                'bias': self.matrix.bias,
            }
        else:
            self.inputs = {'x': self.matrix.input, 'w': self.matrix.weights}

        if self.with_relu:
            activation_type = "relu"
        else:
            activation_type = ""
        self.attrs = {
            'activation_type': activation_type,
            'quant_round_type': self.quant_round_type,
            'quant_max_bound': self.quant_max_bound,
            'quant_min_bound': self.quant_min_bound,
            'scale_in': self.scale_in,
            'scale_weights': self.scale_weights,
        }

        self.outputs = {
            'out': quant_linear_refer(
                self.matrix,
                self.with_bias,
                self.scale_in,
                self.scale_weights,
                self.quant_round_type,
                self.quant_max_bound,
                self.quant_min_bound,
                self.with_relu,
            )
        }

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, check_dygraph=False)


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm(),
    "QuantLinear only supports cuda kernel.",
)
class TestQuantLinearOpNoBias1(TestQuantLinearOp):
    def config(self):
        self.with_bias = False
        self.with_relu = False
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(16, 10, 16, 4, 4, 2)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(
            self.matrix.weights,
            self.scale_weights,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm(),
    "QuantLinear only supports cuda kernel.",
)
class TestQuantLinearOpNoBias2(TestQuantLinearOp):
    def config(self):
        self.with_bias = False
        self.with_relu = False
        self.quant_round_type = 0
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(2, 8, 10, 1, 1, 2)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(
            self.matrix.weights,
            self.scale_weights,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm(),
    "QuantLinear only supports cuda kernel.",
)
class TestQuantLinearOpNoBias3(TestQuantLinearOp):
    def config(self):
        self.with_bias = False
        self.with_relu = False
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(2, 6, 10, 1, 1, 2)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(
            self.matrix.weights,
            self.scale_weights,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm(),
    "QuantLinear only supports cuda kernel.",
)
class TestQuantLinearOpNoBias4(TestQuantLinearOp):
    def config(self):
        self.with_bias = False
        self.with_relu = False
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(2, 14, 10, 1, 1, 2)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(
            self.matrix.weights,
            self.scale_weights,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm(),
    "QuantLinear only supports cuda kernel.",
)
class TestQuantLinearOpWithBias1(TestQuantLinearOp):
    def config(self):
        self.with_bias = True
        self.with_relu = True
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(1, 64, 32, 3, 3, 1)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(
            self.matrix.weights,
            self.scale_weights,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm(),
    "QuantLinear only supports cuda kernel.",
)
class TestQuantLinearOpWithBias2(TestQuantLinearOp):
    def config(self):
        self.with_bias = True
        self.with_relu = True
        self.quant_round_type = 0
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(3, 8, 10, 2, 1, 2)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(
            self.matrix.weights,
            self.scale_weights,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm(),
    "QuantLinear only supports cuda kernel.",
)
class TestQuantLinearOpWithPadding1(TestQuantLinearOp):
    def config(self):
        self.with_bias = True
        self.with_relu = True
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(1, 4, 4, 128, 128, 2)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(
            self.matrix.weights,
            self.scale_weights,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm(),
    "QuantLinear only supports cuda kernel.",
)
class TestQuantLinearOpWithPadding2(TestQuantLinearOp):
    def config(self):
        self.with_bias = True
        self.with_relu = True
        self.quant_round_type = 0
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(1, 4, 3, 128, 128, 2)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(
            self.matrix.weights,
            self.scale_weights,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm(),
    "QuantLinear only supports cuda kernel.",
)
class TestQuantLinearOp_NumFlattenDims_NegOne(unittest.TestCase):
    def test_api(self):
        def run_program(num_flatten_dims):
            with paddle.pir_utils.OldIrGuard():
                paddle.seed(SEED)
                np.random.seed(SEED)
                startup_program = paddle.base.Program()
                main_program = paddle.base.Program()

                with paddle.base.program_guard(main_program, startup_program):
                    quant_round_type = 0
                    quant_max_bound = 127.0
                    quant_min_bound = -127.0

                    input = np.random.random([2, 2, 25]).astype("float32")
                    scale_in = get_scale_in(input)
                    x = paddle.static.data(
                        name="x",
                        shape=[2, 2, 25],
                        dtype="float32",
                    )

                    weight = np.random.random([25, 1]).astype("float32")
                    scale_weight = get_scale_weights(weight)
                    weight = quant_weights(
                        weight,
                        scale_weight,
                        quant_round_type,
                        quant_max_bound,
                        quant_min_bound,
                    )
                    w = paddle.static.data(
                        name="w",
                        shape=[25, 1],
                        dtype="int8",
                    )

                    out = quant_linear(
                        x=x,
                        size=1,
                        num_flatten_dims=num_flatten_dims,
                        w=w,
                        scale_in=scale_in,
                        scale_weight=scale_weight.tolist(),
                        quant_round_type=quant_round_type,
                        quant_max_bound=quant_max_bound,
                        quant_min_bound=quant_min_bound,
                    )

                place = base.CUDAPlace(0)
                exe = base.Executor(place=place)
                exe.run(startup_program)
                out = exe.run(
                    main_program,
                    feed={"x": input, "w": weight},
                    fetch_list=[out],
                )
                return out

        res_1 = run_program(-1)
        res_2 = run_program(2)
        np.testing.assert_array_equal(res_1, res_2)


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm(),
    "QuantLinear only supports cuda kernel.",
)
class TestQuantLinearOpError(unittest.TestCase):
    def test_errors(self):
        with paddle.pir_utils.OldIrGuard():
            with program_guard(Program(), Program()):
                quant_round_type = 0
                quant_max_bound = 127.0
                quant_min_bound = -127.0

                input_data = np.random.random((2, 4)).astype("float32")
                scale_in = get_scale_in(input_data)

                weight = np.random.random([25, 1]).astype("float32")
                scale_weight = get_scale_weights(weight)
                weight = quant_weights(
                    weight,
                    scale_weight,
                    quant_round_type,
                    quant_max_bound,
                    quant_min_bound,
                )

                def test_Variable():
                    with paddle_static_guard():
                        w2 = paddle.static.data(
                            name='w2', shape=[25, 1], dtype='int8'
                        )
                        quant_linear(
                            x=input_data,
                            size=1,
                            num_flatten_dims=1,
                            w=w2,
                            scale_in=scale_in,
                            scale_weight=scale_weight.tolist(),
                            quant_round_type=quant_round_type,
                            quant_max_bound=quant_max_bound,
                            quant_min_bound=quant_min_bound,
                        )

                self.assertRaises(TypeError, test_Variable)

                def test_type():
                    with paddle_static_guard():
                        x2 = paddle.static.data(
                            name='x2', shape=[-1, 4], dtype='int32'
                        )
                        w2 = paddle.static.data(
                            name='w2', shape=[25, 1], dtype='int8'
                        )
                        paddle.static.nn.fc(
                            x=x2,
                            size=1,
                            num_flatten_dims=1,
                            w=w2,
                            scale_in=scale_in,
                            scale_weight=scale_weight.tolist(),
                            quant_round_type=quant_round_type,
                            quant_max_bound=quant_max_bound,
                            quant_min_bound=quant_min_bound,
                        )

                self.assertRaises(TypeError, test_type)

                def test_Variable():
                    with paddle_static_guard():
                        x3 = paddle.static.data(
                            name='x3', shape=[-1, 4], dtype='float32'
                        )
                        quant_linear(
                            x=x3,
                            size=1,
                            num_flatten_dims=1,
                            w=weight,
                            scale_in=scale_in,
                            scale_weight=scale_weight.tolist(),
                            quant_round_type=quant_round_type,
                            quant_max_bound=quant_max_bound,
                            quant_min_bound=quant_min_bound,
                        )

                self.assertRaises(TypeError, test_Variable)

                def test_type():
                    with paddle_static_guard():
                        x3 = paddle.static.data(
                            name='x3', shape=[-1, 4], dtype='float32'
                        )
                        w3 = paddle.static.data(
                            name='w3', shape=[25, 1], dtype='int32'
                        )
                        paddle.static.nn.fc(
                            x=x3,
                            size=1,
                            num_flatten_dims=1,
                            w=w3,
                            scale_in=scale_in,
                            scale_weight=scale_weight.tolist(),
                            quant_round_type=quant_round_type,
                            quant_max_bound=quant_max_bound,
                            quant_min_bound=quant_min_bound,
                        )

                self.assertRaises(TypeError, test_type)

                scale_weight = 1.0

                def test_type():
                    with paddle_static_guard():
                        x4 = paddle.static.data(
                            name='x4', shape=[-1, 4], dtype='float32'
                        )
                        w4 = paddle.static.data(
                            name='w4', shape=[25, 1], dtype='int8'
                        )
                        paddle.static.nn.fc(
                            x=x4,
                            size=1,
                            num_flatten_dims=1,
                            w=w4,
                            scale_in=scale_in,
                            scale_weight=scale_weight,
                            quant_round_type=quant_round_type,
                            quant_max_bound=quant_max_bound,
                            quant_min_bound=quant_min_bound,
                        )

                self.assertRaises(TypeError, test_type)

                scale_weight = []

                def test_param_length():
                    with paddle_static_guard():
                        x4 = paddle.static.data(
                            name='x4', shape=[-1, 4], dtype='float32'
                        )
                        w4 = paddle.static.data(
                            name='w4', shape=[25, 1], dtype='int8'
                        )
                        paddle.static.nn.fc(
                            x=x4,
                            size=1,
                            num_flatten_dims=1,
                            w=w4,
                            scale_in=scale_in,
                            scal=scale_weight,
                            quant_round_type=quant_round_type,
                            quant_max_bound=quant_max_bound,
                            quant_min_bound=quant_min_bound,
                        )

                self.assertRaises(TypeError, test_param_length)


if __name__ == "__main__":
    unittest.main()
