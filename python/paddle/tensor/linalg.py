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
from paddle.common_ops_import import *

# TODO: define functions of linear algebra   
__all__ = [
    'matmul',
    #  'dot',
    #  'einsum',
    'norm',
    #  'transpose',
    #  'dist',
    #  't',
    #  'cross',
    #  'cholesky',
    #  'tensordot'
]


def matmul(x, y, transpose_x=False, transpose_y=False, alpha=1.0, name=None):
    """
    Applies matrix multiplication to two tensors.

    Currently, the input tensors' rank can be any, but when the rank of any
    inputs is bigger than 3, this two inputs' rank should be equal.

    The actual behavior depends on the shapes of :math:`x`, :math:`y` and the
    flag values of :attr:`transpose_x`, :attr:`transpose_y`. Specifically:

    - If a transpose flag is specified, the last two dimensions of the tensor
      are transposed. If the tensor is rank-1 of shape :math:`[D]`, then for
      :math:`x` it is treated as :math:`[1, D]` in nontransposed form and as
      :math:`[D, 1]` in transposed form, whereas for :math:`y` it is the
      opposite: It is treated as :math:`[D, 1]` in nontransposed form and as
      :math:`[1, D]` in transposed form.

    - After transpose, the two tensors are 2-D or n-D and matrix multiplication
      performs in the following way.

      - If both are 2-D, they are multiplied like conventional matrices.
      - If either is n-D, it is treated as a stack of matrices residing in the
        last two dimensions and a batched matrix multiply supporting broadcast
        applies on the two tensors.

    Also note that if the raw tensor :math:`x` or :math:`y` is rank-1 and
    nontransposed, the prepended or appended dimension :math:`1` will be
    removed after matrix multiplication.

    Args:
        x (Variable): The input variable which is a Tensor or LoDTensor.
        y (Variable): The input variable which is a Tensor or LoDTensor.
        transpose_x (bool): Whether to transpose :math:`x` before multiplication.
        transpose_y (bool): Whether to transpose :math:`y` before multiplication.
        alpha (float): The scale of output. Default 1.0.
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Variable: The product Tensor (or LoDTensor) variable.

    Examples:
        .. code-block:: python

            # Examples to clarify shapes of the inputs and output
            # x: [B, ..., M, K], y: [B, ..., K, N]
            # paddle.matmul(x, y)  # out: [B, ..., M, N]

            # x: [B, M, K], y: [B, K, N]
            # paddle.matmul(x, y)  # out: [B, M, N]

            # x: [B, M, K], y: [K, N]
            # paddle.matmul(x, y)  # out: [B, M, N]

            # x: [M, K], y: [K, N]
            # paddle.matmul(x, y)  # out: [M, N]

            # x: [B, M, K], y: [K]
            # paddle.matmul(x, y)  # out: [B, M]

            # x: [K], y: [K]
            # paddle.matmul(x, y)  # out: [1]

            # x: [M], y: [N]
            # paddle.matmul(x, y, True, True)  # out: [M, N]

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[2, 3], dtype='float32')
            y = fluid.data(name='y', shape=[3, 2], dtype='float32')
            out = paddle.matmul(x, y, True, True)
    """
    attrs = {
        'transpose_X': transpose_x,
        'transpose_Y': transpose_y,
        'alpha': float(alpha),
    }

    if in_dygraph_mode():
        return core.ops.matmul(x, y, 'transpose_X', transpose_x, 'transpose_Y',
                               transpose_y, 'alpha', float(alpha))

    def __check_input(x, y):
        var_names = {'x': x, 'y': y}
        for name, val in var_names.items():
            check_variable_and_dtype(
                val, name, ['float16', 'float32', 'float64'], 'matmul')
        x_shape = list(x.shape)
        y_shape = list(y.shape)
        if len(x_shape) == 1:
            x_shape = [1] + x_shape
        if len(y_shape) == 1:
            y_shape = y_shape + [1]

        # check the inner 2 dimensions
        if transpose_x:
            x_shape[-2], x_shape[-1] = x_shape[-1], x_shape[-2]
        if transpose_y:
            y_shape[-2], y_shape[-1] = y_shape[-1], y_shape[-2]
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

    __check_input(x, y)

    helper = LayerHelper('matmul', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='matmul',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs=attrs)
    return out

def norm(input, p='fro', axis=None, keepdim=False, out=None, name=None):
    """
    Returns the matrix norm (Frobenius) or vector norm (the 1-norm, the Euclidean
        or 2-norm, and in general the p-norm for p > 0) of a given tensor.

    Args:
        input(Variable): The input tensor could be N-D tensor, and the input data
            type could be float32 or float64.
        p(int|string, optional): Order of the norm. Supported values are `fro`, `1`, `2`,
            and any positive real number yielding the corresponding p-norm.
        axis(int|list, optional): The axis on which to apply norm operation. If axis is int
            or list with only one element, the vector norm is computed over the axis.
            If axis is a list with two element, the matrix norm is computed over the axis.
            If `axis < 0`, the dimension to norm operation is rank(X) + axis.
       keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have fewer dimension
            than the :attr:`input` unless :attr:`keepdim` is true, default
            value is False.
       name(str, optional): The default value is None.  Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`
    Returns:
        Variable: Tensor, results of norm operation on the specified dim of input tensor,
        it's data type is the same as input's Tensor..
    Raises:
        TypeError, if out data type is different with the input data type.
        ValueError: If `p` or `axis` is invalid.
    """

    def frobenius_norm(input, dim=None, keepdim=False, out=None):
        """
        The frobenius norm OP is to calculate the frobenius norm of certain two dimensions of Tensor `input`.
        Args:
          input (Variable): Tensor, data type float32, float64.
          dim (list, optional): None for last two dimensions.
          keepdim (bool, optional): Whether keep the dimensions as the `input`, Default False.
          out(Variable): The tensor variable storing the output.
        """
        if dim is not None and not (isinstance(dim, list) and len(dim) == 2):
            raise ValueError(
                "The dim of frobenius norm op should be None or two elements list!"
            )
        attrs = {
            'dim': dim if dim != None else [-2, -1],
            'keep_dim': keepdim,
            'reduce_all': False
        }
        check_type_and_dtype(input, 'input', Variable, ['float32', 'float64'],
                             'frobenius_norm')

        helper = LayerHelper('frobenius_norm', **locals())
        if out is None:
            out = helper.create_variable_for_type_inference(
                dtype=helper.input_dtype())
        else:
            check_type(out, 'out', (Variable), 'frobenius_norm')
            if convert_dtype(dtype) != convert_dtype(out.dtype):
                raise TypeError(
                    "In frobenius_norm Op the data type of out must equal the dtype parameter when out is not None"
                )
        helper.append_op(
            type='frobenius_norm',
            inputs={'X': input},
            outputs={'Out': out},
            attrs=attrs)
        return out

    def vector_norm(input, porder=None, axis=None, keepdim=False, out=None):
        """
        Calculate the p-order vector norm for certain  dimension of Tensor `input`.
        Args:
          input (Variable): Tensor, data type float32, float64.
          axis (int, optional): None for last dimension.
          porder (int, optional): None for porder=2.
          keepdim (bool, optional): Whether keep the dimensions as the `input`, Default False.
          out(Variable): The tensor variable storing the output.
        """
        if porder is not None and isinstance(porder, int):
            raise ValueError(
                "The p-oreder of pnorm op (vector norm) should be None or int!")
        if axis is not None and not isinstance(porder, int):
            raise ValueError(
                "The axis of pnorm op (vector norm) should be None or int!")
        attrs = {
            'axis': axis if axis is not None else -1,
            'porder': porder if porder is not None else 2,
            'keepdim': keepdim,
            'epsilon': 1e-12,
        }
        check_type_and_dtype(input, 'input', Variable, ['float32', 'float64'],
                             'p_norm')

        helper = LayerHelper('p_norm', **locals())
        if out is not None:
            out = helper.create_variable_for_type_inference(
                dtype=helper.input_dtype())
        else:
            check_type(out, 'out', (Variable), 'p_norm')
            if convert_dtype(dtype) != convert_dtype(out.dtype):
                raise TypeError(
                    "In p_norm Op the data type of out must equal the dtype parameter when out is not None"
                )

        helper.append_op(
            type='p_norm',
            inputs={'X': input},
            outputs={'Out': out},
            attrs=attrs)
        return out

    if axis is None and p is not None:
        if isinstance(p, str):
            if p == "fro":
                return frobenius_norm(
                    input, axis=axis, keepdim=keepdim, out=out)
            else:
                raise ValueError(
                    "only valid string values are 'fro', found {}".format(p))
        if not isinstance(p, str):
            if isinstance(p, int):
                return vector_norm(input, axis=axis, porder=p, keedim=keepdim)
            else:
                raise ValueError("only valid p type is string or int, found {}".
                                 format(type(p)))

    if isinstance(axis, list) and len(axis) == 1:
        axis = axis[0]

    #calculate vector norm, where axis is int or list with only one integer
    if isinstance(axis, int):
        if isinstance(p, int):
            return vector_norm(input, axis=axis, porder=p, keedim=keepdim)
        else:
            raise ValueError(
                "unspport p for p-order vector norm. except integer, found {}".
                format(axis))
    #calculate matrix norm, where axis is list with two integers
    elif isinstance(axis, list) and len(list) == 2:
        if p == "fro":
            return frobenius_norm(input, axis=axis, keepdim=keepdim, out=out)
        else:
            raise ValueError(
                "unspport p for matrix norm, expcept 'fro', found {}".format(p))
    else:
        raise ValueError(
            "except axis type int or list (length of list <=2), found {}".
            formar(axis))
