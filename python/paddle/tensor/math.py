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

# TODO: define math functions  
__all__ = [#'abs',
           # 'acos',
           # 'asin',
           # 'atan',
           # 'ceil',
           # 'cos',
           # 'cumsum',
           # 'elementwise_add',
           # 'elementwise_div',
           # 'elementwise_floordiv',
           # 'elementwise_max',
           # 'elementwise_min',
           # 'elementwise_mod',
           # 'elementwise_mul',
           # 'elementwise_pow',
           # 'elementwise_sub',
           # 'exp',
           # 'floor',
           # 'increment',
           # 'log',
           # 'mul',
           # 'multiplex',
           # 'pow',
           # 'reciprocal',
           # 'reduce_max',
           # 'reduce_min',
           # 'reduce_prod',
           # 'reduce_sum',
           # 'round',
           # 'rsqrt',
           # 'scale',
           # 'sign',
           # 'sin',
           # 'sqrt',
           # 'square',
           # 'stanh',
           # 'sum',
           # 'sums',
           # 'tanh',
           'elementwise_sum',
           # 'max',
           # 'min',
           'mm',
           # 'div',
           # 'add',
           # 'atan',
           # 'logsumexp',
           # 'inverse',
           # 'log1p',
           # 'erf',
           # 'addcmul',
           # 'addmm'
           ]
from paddle.fluid.layers.layer_function_generator import templatedoc
from paddle.common_ops_import import *


@templatedoc(op_type="sum")
def elementwise_sum(inputs, name=None):
    """
    ${comment}
    
    Case 1:
    ::
        Input:
            Input. Shape = [2, 3]
            Input = [[1, 2, 3],
                     [4, 5, 6]]

        Output:
            The output. Shape = [2, 3]
            Output = [[1, 2, 3],
                      [4, 5, 6]]

    Case 2:
    ::
        Input:
            First input:
            Input1. Shape = [2, 3]
            Input1 = [[1, 2, 3],
                      [4, 5, 6]]

        The second input:
            Input2. Shape = [2, 3]
            Input2 = [[7, 8, 9],
                      [10, 11, 12]]

        Output:
            The output. Shape = [2, 3]
            Output = [[8, 10, 12],
                      [14, 16, 18]]

    Args:
        x (Variable|list(Variable)): ${x_comment}

    Returns:
        Variable: ${out_comment}

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            input0 = fluid.layers.fill_constant(shape=[2, 3], dtype='int64', value=5)
            input1 = fluid.layers.fill_constant(shape=[2, 3], dtype='int64', value=3)
            sum = paddle.elementwise_sum([input0, input1])

            # You can print out 'sum' via executor.
            out = fluid.layers.Print(sum, message="the sum of input0 and input1: ")
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_main_program())

            # The printed result is:
            # 1570701754	the sum of input0 and input1: 	The place is:CPUPlace
            # Tensor[elementwise_sum_0.tmp_0]
            #    shape: [2,3,]
            #    dtype: l
            #    data: 8,8,8,8,8,8,

            # the sum of input0 and input1 is 2-D Tensor with shape [2,3].
            # dtype is the corresponding C++ data type, which may vary in different environments.
            # Eg: if the data type of tensor is int64, then the corresponding C++ data type is int64_t, 
            #       so the dtype value is typeid(int64_t).Name(), which is 'x' on MacOS, 'l' on Linux, 
            #       and '__int64' on Windows. They both represent 64-bit integer variables.
    """

    helper = LayerHelper('elementwise_sum', **locals())
    out = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype('x'))
    helper.append_op(
        type='sum',
        inputs={'X': inputs},
        outputs={'Out': out},
        attrs={'use_mkldnn': False})

    return out


def mm(input, mat2, out=None, name=None):
    """
    Applies matrix multiplication to two tensors.

    Currently, the input tensors' rank can be any, but when the rank of any
    inputs is bigger than 3, this two inputs' rank should be equal.


    Also note that if the raw tensor :math:`x` or :math:`y` is rank-1 and
    nontransposed, the prepended or appended dimension :math:`1` will be
    removed after matrix multiplication.

    Args:
        x (Variable): The input variable which is a Tensor or LoDTensor.
        y (Variable): The input variable which is a Tensor or LoDTensor.
        out(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: The product Tensor (or LoDTensor) variable.

    Examples:
        .. code-block:: python

            # Examples to clarify shapes of the inputs and output
            # x: [B, ..., M, K], y: [B, ..., K, N]
            # fluid.layers.matmul(x, y)  # out: [B, ..., M, N]

            # x: [B, M, K], y: [B, K, N]
            # fluid.layers.matmul(x, y)  # out: [B, M, N]

            # x: [B, M, K], y: [K, N]
            # fluid.layers.matmul(x, y)  # out: [B, M, N]

            # x: [M, K], y: [K, N]
            # fluid.layers.matmul(x, y)  # out: [M, N]

            # x: [B, M, K], y: [K]
            # fluid.layers.matmul(x, y)  # out: [B, M]

            # x: [K], y: [K]
            # fluid.layers.matmul(x, y)  # out: [1]

            import paddle
            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[2, 3], dtype='float32')
            y = fluid.data(name='y', shape=[3, 2], dtype='float32')
            out = paddle.mm(x, y) # out shape is [2, 2]
    """
    if in_dygraph_mode():
        return core.ops.matmul(x, y)

    def __check_input(x, y):
        var_names = {'x': x, 'y': y}
        for name, val in var_names.items():
            check_variable_and_dtype(val, name,
                                     ['float16', 'float32', 'float64'], 'mm')
        x_shape = list(x.shape)
        y_shape = list(y.shape)
        if len(x_shape) == 1:
            x_shape = [1] + x_shape
        if len(y_shape) == 1:
            y_shape = y_shape + [1]

        # check the inner 2 dimensions
        if x_shape[-1] != y_shape[-2]:
            assert (x_shape[-1] == -1) or (y_shape[-2] == -1),                         \
                "After performing an optional transpose, Input X's width should be "   \
                "equal to Y's width for multiplication "                               \
                "prerequisites. But received X's shape: %s, Y's shape: %s\n" %         \
                (x_shape, y_shape)

        if len(y_shape) > 2 and len(x_shape) > 2:
            for i, dim_x in enumerate(x_shape[:-2]):
                # don't check neg shape
                if dim_x < 0 or y_shape[i] < 0:
                    continue
                if dim_x != y_shape[i]:
                    raise ValueError(
                        "When the matrix is larger than 2 dimensions, the higher "
                        "dimensional values of the two matrices need to be equal. "
                        "But received x_shape[%d] != y_shape[%d]. X's shape: %s, "
                        "Y's shape: %s.\n" % (i, i, x_shape, y_shape))

    __check_input(input, mat2)

    helper = LayerHelper('mm', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='matmul', inputs={'X': input,
                               'Y': mat2}, outputs={'Out': out})
    return out
