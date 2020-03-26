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
__all__ = [# 'abs',
#            'acos',
#            'asin',
#            'atan',
#            'ceil',
#            'cos',
#            'cumsum',
#            'elementwise_add',
#            'elementwise_div',
#            'elementwise_floordiv',
#            'elementwise_max',
#            'elementwise_min',
#            'elementwise_mod',
#            'elementwise_mul',
#            'elementwise_pow',
#            'elementwise_sub',
#            'exp',
#            'floor',
#            'increment',
#            'log',
            'mul',
#            'multiplex',
            'pow',
#            'reciprocal',
#            'reduce_max',
#            'reduce_min',
#            'reduce_prod',
#            'reduce_sum',
#            'round',
#            'rsqrt',
#            'scale',
#            'sign',
#            'sin',
#            'sqrt',
#            'square',
#            'stanh',
#            'sum',
#            'sums',
#            'tanh',
#            'elementwise_sum',
#            'max',
#            'min',
#            'mm',
#            'div',
#            'add',
#            'atan',
#            'logsumexp',
#            'inverse',
#            'log1p',
#            'erf',
#            'addcmul',
#            'addmm'
            ]

from paddle.common_ops_import import *
from paddle.fluid.layers.layer_function_generator import templatedoc



@templatedoc()
def pow(input, exponent, out=None, name=None):
    """
    This is Pow Activation Operator.

    :math:`out = input^{exponent}`

    Args:
        input(Variable): A ``Tensor`` or ``LoDTensor`` . The data type is ``float32`` or ``float64``.
        exponent(float32|Variable): A scalar with type ``float32`` or a ``Tensor`` with shape [1] and type ``float32``.
        out (Variable, optional):  The Variable that stores results of the operation. If out is None, a new Varibale will be create to store the results.
        name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: A ``Tensor`` or ``LoDTensor``. The data type is same as ``input``.

    Examples:

        .. code-block:: python

            import paddle

            x = paddle.fluid.data(name="x", shape=[32,32], dtype="float32")

            # example 1: argument exponent is float
            res = paddle.fluid.data(name="output", shape=[32,32], dtype="float32")
            y_1 = paddle.tensor.pow(x, 2.0, out=res)
            # y_1 is x^{2.0}

            # example 2: argument exponent is Variable
            exponet_tensor = fluid.layers.fill_constant([1], "float32", 3.0)
            res = paddle.fluid.data(name="output", shape=[32,32], dtype="float32")
            y_2 = fluid.layers.pow(x, exponent_tensor, out=res)
            # y_2 is x^{3.0}
    """
    helper = LayerHelper('pow', **locals())
    inputs = {'X': input}
    attrs = {}
    if isinstance(exponent, Variable):
        exponent.stop_gradient = True
        inputs['FactorTensor'] = exponent
    else:
        attrs['factor'] = exponent

    if out is None:
        if name:
            out =  helper.create_variable(name=name, dtype=input.dtype, persistable=False)
        else:
            out = helper.create_variable_for_type_inference(dtype=input.dtype)
    else:
        check_dtype(
            out.dtype, out.name,
            convert_dtype(input.dtype), 'pow',
            '(The out data type in pow must be the same with input data type.)'
        )
        if name:
            warning.warn("The output Variable name of the paddle.tensor.pow operation can only be given by parameter out or name. When parameter out and name are set at the same time, out has a higher priority than name. Finally, the output Variable name is same as the out name %s" % out.name, category=UserWarning,stacklevel=2)

    helper.append_op(
        type='pow', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out

def mul(x, y, x_num_col_dims=1, y_num_col_dims=1, out=None, name=None):
    """
    Mul Operator.
    This operator is used to perform matrix multiplication for input $x$ and $y$.
    The equation is:

    ..  math::
        Out = x * y

    Both the input $x$ and $y$ can carry the LoD (Level of Details) information, or not. But the output only shares the LoD information with input $x$.

    Args:
        x (Variable): The first input Tensor/LoDTensor of mul_op.
        y (Variable): The second input Tensor/LoDTensor of mul_op.
        x_num_col_dims (int, optional): The mul_op can take tensors with more than two dimensions as its inputs. If the input $x$ is a tensor with more than two dimensions, $x$ will be flattened into a two-dimensional matrix first. The flattening rule is: the first `num_col_dims` will be flattened to form the first dimension of the final matrix (the height of the matrix), and the rest `rank(x) - num_col_dims` dimensions are flattened to form the second dimension of the final matrix (the width of the matrix). As a result, height of the flattened matrix is equal to the product of $x$'s first `x_num_col_dims` dimensions' sizes, and width of the flattened matrix is equal to the product of $x$'s last `rank(x) - num_col_dims` dimensions' size. For example, suppose $x$ is a 6-dimensional tensor with the shape [2, 3, 4, 5, 6], and `x_num_col_dims` = 3. Thus, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24, 30]. Default is 1. 
        y_num_col_dims (int, optional): The mul_op can take tensors with more than two dimensions as its inputs. If the input $y$ is a tensor with more than two dimensions, $y$ will be flattened into a two-dimensional matrix first. The attribute `y_num_col_dims` determines how $y$ is flattened. See comments of `x_num_col_dims` for more details. Default is 1. 
        out(Variable, optinal):  The Variable that stores results of the operation. If out is None, a new Varibale will be create to store the results.
        name (str, optional): Name of the output. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`. Default is None. If both of out and name are not None, the output name will be same as out. 

    Returns:
        Variable(Tensor/LoDTensor): The output Tensor/LoDTensor of mul op.

    Examples:
        ..  code-block:: python
            
            import paddle
            dataX = paddle.fluid.layers.data(name="dataX", append_batch_size = False, shape=[2, 5], dtype="float32")
            dataY = paddle.fluid.layers.data(name="dataY", append_batch_size = False, shape=[5, 3], dtype="float32")
            
            res = paddle.fluid.layers.data(name="output", append_batch_size = False, shape=[2, 3], dtype="float32")
            output = paddle.fluid.layers.mul(dataX, dataY,
                                      x_num_col_dims = 1,
                                      y_num_col_dims = 1, 
                                      out=res)
            

    """
    inputs = {"X": [x], "Y": [y]}
    attrs = {"x_num_col_dims": x_num_col_dims, "y_num_col_dims": y_num_col_dims}
    if in_dygraph_mode():
        outs = core.ops.mul(inputs, attrs)
        return outs['Out'][0]

    helper = LayerHelper("mul", **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'mul')
    check_variable_and_dtype(y, 'y', ['float16', 'float32', 'float64'], 'mul')

    if out is None:
        if name:
            out =  helper.create_variable(name=name, dtype=x.dtype, persistable=False)
        else:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        check_dtype(
            out.dtype, out.name,
            convert_dtype(x.dtype), 'mul',
            '(The out data type in pow must be the same with input data type.)'
        )
        if name:
            warning.warn("The output Variable name of the paddle.tensor.pow operation can only be given by parameter out or name. When parameter out and name are set at the same time, out has a higher priority than name. Finally, the output Variable name is same as the out name %s" % out.name, category=UserWarning,stacklevel=2)
    helper.append_op(
        type="mul", inputs={"X": x,
                            "Y": y}, attrs=attrs, outputs={"Out": out})
    return out

