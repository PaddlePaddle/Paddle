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

import numpy as np

import paddle
from paddle import _C_ops
from paddle.common_ops_import import VarDesc
from paddle.utils.inplace_utils import inplace_apis_in_dygraph_only

from ..base.data_feeder import check_dtype, check_type, check_variable_and_dtype
from ..common_ops_import import Variable
from ..framework import LayerHelper, in_dynamic_mode, in_dynamic_or_pir_mode
from .creation import full
from .manipulation import cast
from .math import _get_reduce_axis

__all__ = []

# Consistent with kDefaultDim from C++ Backend
K_DEFAULT_DIM = 9


def transpose(x, perm, name=None):
    """
    Permute the data dimensions of `input` according to `perm`.

    The `i`-th dimension  of the returned tensor will correspond to the
    perm[i]-th dimension of `input`.

    Args:
        x (Tensor): The input Tensor. It is a N-D Tensor of data types bool, float32, float64, int32.
        perm (list|tuple): Permute the input according to the data of perm.
        name (str, optional): The name of this layer. For more information, please refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        Tensor: A transposed n-D Tensor, with data type being bool, float32, float64, int32, int64.

    Examples:

        .. code-block:: text

            x = [[[ 1  2  3  4] [ 5  6  7  8] [ 9 10 11 12]]
                 [[13 14 15 16] [17 18 19 20] [21 22 23 24]]]
            shape(x) =  [2,3,4]

            # Example 1
            perm0 = [1,0,2]
            y_perm0 = [[[ 1  2  3  4] [13 14 15 16]]
                       [[ 5  6  7  8]  [17 18 19 20]]
                       [[ 9 10 11 12]  [21 22 23 24]]]
            shape(y_perm0) = [3,2,4]

            # Example 2
            perm1 = [2,1,0]
            y_perm1 = [[[ 1 13] [ 5 17] [ 9 21]]
                       [[ 2 14] [ 6 18] [10 22]]
                       [[ 3 15]  [ 7 19]  [11 23]]
                       [[ 4 16]  [ 8 20]  [12 24]]]
            shape(y_perm1) = [4,3,2]

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.randn([2, 3, 4])
            >>> x_transposed = paddle.transpose(x, perm=[1, 0, 2])
            >>> print(x_transposed.shape)
            [3, 2, 4]

    """
    if in_dynamic_or_pir_mode():
        return _C_ops.transpose(x, perm)
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'bool',
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'uint16',
                'complex64',
                'complex128',
            ],
            'transpose',
        )
        check_type(perm, 'perm', (list, tuple), 'transpose')
        if isinstance(perm, tuple):
            perm = list(perm)
        if len(perm) != len(x.shape):
            raise ValueError(
                "Input(perm) is the permutation of dimensions of Input(x), "
                "its length should be equal to dimensions of Input(x), "
                f"but received dimension of Input(x) is {len(x.shape)}, "
                f"the length of Input(perm) is {len(perm)}."
            )
        for idx, dim in enumerate(perm):
            if dim >= len(x.shape):
                raise ValueError(
                    "Each element in Input(perm) should be less than Input(x)'s dimension, "
                    "but %d-th element in Input(perm) is %d which exceeds Input(x)'s "
                    "dimension %d." % (idx, perm[idx], len(x.shape))
                )

        helper = LayerHelper('transpose', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        x_shape = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='transpose2',
            inputs={'X': [x]},
            outputs={'Out': [out], 'XShape': [x_shape]},
            attrs={'axis': perm},
        )
        return out


@inplace_apis_in_dygraph_only
def transpose_(x, perm, name=None):
    r"""
    Inplace version of ``transpose`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_transpose`.
    """
    if in_dynamic_mode():
        return _C_ops.transpose_(x, perm)


def matmul(x, y, transpose_x=False, transpose_y=False, name=None):
    """
    Applies matrix multiplication to two tensors. `matmul` follows
    the complete broadcast rules,
    and its behavior is consistent with `np.matmul`.

    Currently, the input tensors' number of dimensions can be any, `matmul` can be used to
    achieve the `dot`, `matmul` and `batchmatmul`.

    The actual behavior depends on the shapes of :math:`x`, :math:`y` and the
    flag values of :attr:`transpose_x`, :attr:`transpose_y`. Specifically:

    - If a transpose flag is specified, the last two dimensions of the tensor
      are transposed. If the tensor is ndim-1 of shape, the transpose is invalid. If the tensor
      is ndim-1 of shape :math:`[D]`, then for :math:`x` it is treated as :math:`[1, D]`, whereas
      for :math:`y` it is the opposite: It is treated as :math:`[D, 1]`.

    The multiplication behavior depends on the dimensions of `x` and `y`. Specifically:

    - If both tensors are 1-dimensional, the dot product result is obtained.

    - If both tensors are 2-dimensional, the matrix-matrix product is obtained.

    - If the `x` is 1-dimensional and the `y` is 2-dimensional,
      a `1` is prepended to its dimension in order to conduct the matrix multiply.
      After the matrix multiply, the prepended dimension is removed.

    - If the `x` is 2-dimensional and `y` is 1-dimensional,
      the matrix-vector product is obtained.

    - If both arguments are at least 1-dimensional and at least one argument
      is N-dimensional (where N > 2), then a batched matrix multiply is obtained.
      If the first argument is 1-dimensional, a 1 is prepended to its dimension
      in order to conduct the batched matrix multiply and removed after.
      If the second argument is 1-dimensional, a 1 is appended to its
      dimension for the purpose of the batched matrix multiple and removed after.
      The non-matrix (exclude the last two dimensions) dimensions are
      broadcasted according the broadcast rule.
      For example, if input is a (j, 1, n, m) tensor and the other is a (k, m, p) tensor,
      out will be a (j, k, n, p) tensor.

    Args:
        x (Tensor): The input tensor which is a Tensor.
        y (Tensor): The input tensor which is a Tensor.
        transpose_x (bool, optional): Whether to transpose :math:`x` before multiplication. Default is False.
        transpose_y (bool, optional): Whether to transpose :math:`y` before multiplication. Default is False.
        name (str, optional): If set None, the layer will be named automatically. For more information, please refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        Tensor: The output Tensor.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> # vector * vector
            >>> x = paddle.rand([10])
            >>> y = paddle.rand([10])
            >>> z = paddle.matmul(x, y)
            >>> print(z.shape)
            []

            >>> # matrix * vector
            >>> x = paddle.rand([10, 5])
            >>> y = paddle.rand([5])
            >>> z = paddle.matmul(x, y)
            >>> print(z.shape)
            [10]

            >>> # batched matrix * broadcasted vector
            >>> x = paddle.rand([10, 5, 2])
            >>> y = paddle.rand([2])
            >>> z = paddle.matmul(x, y)
            >>> print(z.shape)
            [10, 5]

            >>> # batched matrix * batched matrix
            >>> x = paddle.rand([10, 5, 2])
            >>> y = paddle.rand([10, 2, 5])
            >>> z = paddle.matmul(x, y)
            >>> print(z.shape)
            [10, 5, 5]

            >>> # batched matrix * broadcasted matrix
            >>> x = paddle.rand([10, 1, 5, 2])
            >>> y = paddle.rand([1, 3, 2, 5])
            >>> z = paddle.matmul(x, y)
            >>> print(z.shape)
            [10, 3, 5, 5]

    """
    if in_dynamic_or_pir_mode():
        return _C_ops.matmul(x, y, transpose_x, transpose_y)
    else:
        attrs = {
            'trans_x': transpose_x,
            'trans_y': transpose_y,
        }

        def __check_input(x, y):
            var_names = {'x': x, 'y': y}
            for name, val in var_names.items():
                check_variable_and_dtype(
                    val,
                    name,
                    [
                        'int8',
                        'uint16',
                        'float16',
                        'float32',
                        'float64',
                        'complex64',
                        'complex128',
                    ],
                    'matmul',
                )

        __check_input(x, y)

        helper = LayerHelper('matmul_v2', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='matmul_v2',
            inputs={'X': x, 'Y': y},
            outputs={'Out': out},
            attrs=attrs,
        )
        return out


def norm(x, p='fro', axis=None, keepdim=False, name=None):
    """

    Returns the matrix norm (Frobenius) or vector norm (the 1-norm, the Euclidean
    or 2-norm, and in general the p-norm for p > 0) of a given tensor.

    Note:
        This norm API is different from `numpy.linalg.norm`.
        This api supports high-order input tensors (rank >= 3), and certain axis need to be pointed out to calculate the norm.
        But `numpy.linalg.norm` only supports 1-D vector or 2-D matrix as input tensor.
        For p-order matrix norm, this api actually treats matrix as a flattened vector to calculate the vector norm, NOT REAL MATRIX NORM.

    Args:
        x (Tensor): The input tensor could be N-D tensor, and the input data
            type could be float32 or float64.
        p (float|string, optional): Order of the norm. Supported values are `fro`, `0`, `1`, `2`,
            `inf`, `-inf` and any positive real number yielding the corresponding p-norm. Not supported: ord < 0 and nuclear norm.
            Default value is `fro`.
        axis (int|list|tuple, optional): The axis on which to apply norm operation. If axis is int
            or list(int)/tuple(int)  with only one element, the vector norm is computed over the axis.
            If `axis < 0`, the dimension to norm operation is rank(input) + axis.
            If axis is a list(int)/tuple(int) with two elements, the matrix norm is computed over the axis.
            Default value is `None`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have fewer dimension
            than the :attr:`input` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: results of norm operation on the specified axis of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.arange(24, dtype="float32").reshape([2, 3, 4]) - 12
            >>> print(x)
            Tensor(shape=[2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[-12., -11., -10., -9. ],
              [-8. , -7. , -6. , -5. ],
              [-4. , -3. , -2. , -1. ]],
             [[ 0. ,  1. ,  2. ,  3. ],
              [ 4. ,  5. ,  6. ,  7. ],
              [ 8. ,  9. ,  10.,  11.]]])

            >>> # compute frobenius norm along last two dimensions.
            >>> out_fro = paddle.linalg.norm(x, p='fro', axis=[0,1])
            >>> print(out_fro)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [17.43559647, 16.91153526, 16.73320007, 16.91153526])

            >>> # compute 2-order vector norm along last dimension.
            >>> out_pnorm = paddle.linalg.norm(x, p=2, axis=-1)
            >>> print(out_pnorm)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[21.11871147, 13.19090557, 5.47722578 ],
             [3.74165750 , 11.22497177, 19.13112640]])

            >>> # compute 2-order  norm along [0,1] dimension.
            >>> out_pnorm = paddle.linalg.norm(x, p=2, axis=[0,1])
            >>> print(out_pnorm)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [17.43559647, 16.91153526, 16.73320007, 16.91153526])

            >>> # compute inf-order  norm
            >>> out_pnorm = paddle.linalg.norm(x, p=float("inf"))
            >>> print(out_pnorm)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            12.)

            >>> out_pnorm = paddle.linalg.norm(x, p=float("inf"), axis=0)
            >>> print(out_pnorm)
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[12., 11., 10., 9. ],
             [8. , 7. , 6. , 7. ],
             [8. , 9. , 10., 11.]])

            >>> # compute -inf-order  norm
            >>> out_pnorm = paddle.linalg.norm(x, p=-float("inf"))
            >>> print(out_pnorm)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.)

            >>> out_pnorm = paddle.linalg.norm(x, p=-float("inf"), axis=0)
            >>> print(out_pnorm)
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 1., 2., 3.],
             [4., 5., 6., 5.],
             [4., 3., 2., 1.]])
    """

    def frobenius_norm(input, dim=None, keepdim=False, name=None):
        """
        The frobenius norm OP is to calculate the frobenius norm of certain two dimensions of Tensor `input`.
        Args:
          input (Variable): Tensor, data type float32, float64.
          dim (list, optional): None for last two dimensions. Default None.
          keepdim (bool, optional): Whether keep the dimensions as the `input`, Default False.
          name (str, optional): The default value is None. Normally there is no need for
              user to set this property. For more information, please refer to :ref:`api_guide_Name`.
        """
        if dim is not None and not (isinstance(dim, list) and len(dim) == 2):
            raise ValueError(
                "The dim of frobenius norm op should be None or two elements list!"
            )

        if in_dynamic_mode():
            if dim is None:
                return _C_ops.frobenius_norm(input, [], keepdim, True)
            return _C_ops.frobenius_norm(input, dim, keepdim, False)
        else:
            attrs = {'dim': dim, 'keep_dim': keepdim, 'reduce_all': False}
            if dim is None:
                attrs['reduce_all'] = True
            check_variable_and_dtype(
                input, 'input', ['float32', 'float64'], 'frobenius_norm'
            )

            helper = LayerHelper('frobenius_norm', **locals())
            out = helper.create_variable_for_type_inference(
                dtype=helper.input_dtype()
            )

            helper.append_op(
                type='frobenius_norm',
                inputs={'X': input},
                outputs={'Out': out},
                attrs=attrs,
            )
            return out

    def vector_norm(
        input, porder=None, axis=None, keepdim=False, asvector=False, name=None
    ):
        """
        Calculate the p-order vector norm for certain  dimension of Tensor `input`.
        Args:
          input (Variable): Tensor, data type float32, float64.
          porder (float, optional): None for porder=2.0. Default None.
          axis (int, optional): None for last dimension. Default None.
          keepdim (bool, optional): Whether keep the dimensions as the `input`, Default False.
          asvector (bool, optional): Whether keep the result as a vector, Default False.
          name (str, optional): The default value is None. Normally there is no need for
              user to set this property. For more information, please refer to :ref:`api_guide_Name`.
        """
        if in_dynamic_mode():
            if axis is None:
                axis = -1
            return _C_ops.p_norm(input, porder, axis, 1e-12, keepdim, asvector)
        else:
            if porder is not None:
                check_type(porder, 'porder', (float, int), 'p_norm')
            if axis is not None:
                check_type(axis, 'axis', (int), 'p_norm')
            check_variable_and_dtype(
                input,
                'input',
                ['float16', 'uint16', 'float32', 'float64'],
                'p_norm',
            )

            attrs = {
                'axis': axis if axis is not None else -1,
                'porder': float(porder) if porder is not None else 2.0,
                'keepdim': keepdim,
                'asvector': asvector,
                'epsilon': 1e-12,
            }
            helper = LayerHelper('p_norm', **locals())
            out = helper.create_variable_for_type_inference(
                dtype=helper.input_dtype()
            )

            helper.append_op(
                type='p_norm',
                inputs={'X': input},
                outputs={'Out': out},
                attrs=attrs,
            )
            return out

    def inf_norm(
        input, porder=None, axis=axis, keepdim=False, asvector=False, name=None
    ):
        if in_dynamic_mode():
            out = _C_ops.abs(input)
            if porder == np.float64('inf'):
                return _C_ops.max(out, axis, keepdim)
            else:
                return _C_ops.min(out, axis, keepdim)
        else:
            helper = LayerHelper('inf_norm', **locals())
            out = helper.create_variable_for_type_inference(
                dtype=helper.input_dtype()
            )
            helper.append_op(
                type='abs', inputs={'X': input}, outputs={'Out': out}
            )
            reduce_out = helper.create_variable_for_type_inference(
                dtype=helper.input_dtype()
            )
            reduce_all, axis = _get_reduce_axis(axis, x)
            reduce_type = (
                'reduce_max' if porder == np.float64('inf') else 'reduce_min'
            )
            helper.append_op(
                type=reduce_type,
                inputs={'X': out},
                outputs={'Out': reduce_out},
                attrs={
                    'dim': axis,
                    'keep_dim': keepdim,
                    'reduce_all': reduce_all,
                },
            )

            return reduce_out

    def p_matrix_norm(input, porder=1.0, axis=axis, keepdim=False, name=None):
        """
        NOTE:
            This function actually treats the matrix as flattened vector to calculate vector norm instead of matrix norm.
        """
        if in_dynamic_mode():
            abs_out = _C_ops.abs(input)
            pow_out = _C_ops.pow(abs_out, porder)
            sum_out = _C_ops.sum(pow_out, axis, None, keepdim)
            out = _C_ops.pow(sum_out, float(1.0 / porder))
            return out

        block = LayerHelper('norm', **locals())
        out = block.create_variable_for_type_inference(
            dtype=block.input_dtype()
        )
        abs_out = block.create_variable_for_type_inference(
            dtype=block.input_dtype()
        )
        block.append_op(
            type='abs', inputs={'X': input}, outputs={'Out': abs_out}
        )
        pow_out = block.create_variable_for_type_inference(
            dtype=block.input_dtype()
        )

        block.append_op(
            type='pow',
            inputs={'X': abs_out},
            outputs={'Out': pow_out},
            attrs={'factor': porder},
        )
        sum_out = block.create_variable_for_type_inference(
            dtype=block.input_dtype()
        )
        reduce_all, axis = _get_reduce_axis(axis, x)
        block.append_op(
            type='reduce_sum',
            inputs={'X': pow_out},
            outputs={'Out': sum_out},
            attrs={
                'dim': axis,
                'keep_dim': keepdim,
                'reduce_all': reduce_all,
            },
        )
        block.append_op(
            type='pow',
            inputs={'X': sum_out},
            outputs={'Out': out},
            attrs={'factor': float(1.0 / porder)},
        )
        return out

    if axis is None and p is not None:
        if isinstance(p, str):
            if p == "fro":
                return frobenius_norm(x, dim=axis, keepdim=keepdim, name=name)
            else:
                raise ValueError(
                    f"only valid string values are 'fro', found {p}"
                )
        elif isinstance(p, (int, float)):
            return vector_norm(
                x,
                porder=p,
                axis=axis,
                keepdim=keepdim,
                asvector=True,
                name=name,
            )
        else:
            raise ValueError(
                f"only valid p type is string or float, found {type(p)}"
            )

    if isinstance(axis, tuple):
        axis = list(axis)
    if isinstance(axis, list) and len(axis) == 1:
        axis = axis[0]

    # calculate vector norm, where axis is int or list with only one integer
    if isinstance(axis, int):
        if isinstance(p, str):
            if p == "fro":
                return vector_norm(
                    x,
                    porder=2,
                    axis=axis,
                    keepdim=keepdim,
                    asvector=False,
                    name=name,
                )

            else:
                raise ValueError(
                    f"only valid string values are 'fro', found {p}"
                )
        elif isinstance(p, (int, float)):
            return vector_norm(
                x,
                axis=axis,
                porder=p,
                keepdim=keepdim,
                asvector=False,
                name=name,
            )
        else:
            raise ValueError(
                f"unspport p for p-order vector norm. except float, found {p}"
            )
    # calculate matrix norm, where axis is list with two integers
    elif isinstance(axis, list) and len(axis) == 2:
        if p == "fro":
            return frobenius_norm(x, dim=axis, keepdim=keepdim, name=name)
        elif p == np.inf or p == -np.inf:
            return inf_norm(x, porder=p, axis=axis, keepdim=keepdim, name=name)
        elif p == 0:
            raise ValueError(
                "just support axis type int or list (length of list <=1) if p = 0, found {}".format(
                    axis
                )
            )
        else:
            return p_matrix_norm(
                x, porder=p, axis=axis, keepdim=keepdim, name=name
            )
    else:
        raise ValueError(
            f"except axis type int or list (length of list <=2), found {axis}"
        )


def dist(x, y, p=2, name=None):
    r"""

    Returns the p-norm of (x - y). It is not a norm in a strict sense, only as a measure
    of distance. The shapes of x and y must be broadcastable. The definition is as follows, for
    details, please refer to the `Introduction to Tensor <../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor>`_:

    - Each input has at least one dimension.
    - Match the two input dimensions from back to front, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

    Where, z = x - y, the shapes of x and y are broadcastable, then the shape of z can be
    obtained as follows:

    1. If the number of dimensions of x and y are not equal, prepend 1 to the dimensions of the
    tensor with fewer dimensions.

    For example, The shape of x is [8, 1, 6, 1], the shape of y is [7, 1, 5], prepend 1 to the
    dimension of y.

    x (4-D Tensor):  8 x 1 x 6 x 1

    y (4-D Tensor):  1 x 7 x 1 x 5

    2. Determine the size of each dimension of the output z: choose the maximum value from the
    two input dimensions.

    z (4-D Tensor):  8 x 7 x 6 x 5

    If the number of dimensions of the two inputs are the same, the size of the output can be
    directly determined in step 2. When p takes different values, the norm formula is as follows:

    When p = 0, defining $0^0=0$, the zero-norm of z is simply the number of non-zero elements of z.

    .. math::

        ||z||_{0}=\lim_{p \\rightarrow 0}\sum_{i=1}^{m}|z_i|^{p}

    When p = inf, the inf-norm of z is the maximum element of the absolute value of z.

    .. math::

        ||z||_\infty=\max_i |z_i|

    When p = -inf, the negative-inf-norm of z is the minimum element of the absolute value of z.

    .. math::

        ||z||_{-\infty}=\min_i |z_i|

    Otherwise, the p-norm of z follows the formula,

    .. math::

        ||z||_{p}=(\sum_{i=1}^{m}|z_i|^p)^{\\frac{1}{p}}

    Args:
        x (Tensor): 1-D to 6-D Tensor, its data type is bfloat16, float16, float32 or float64.
        y (Tensor): 1-D to 6-D Tensor, its data type is bfloat16, float16, float32 or float64.
        p (float, optional): The norm to be computed, its data type is float32 or float64. Default: 2.
        name (str, optional): The default value is `None`. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Tensor that is the p-norm of (x - y).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[3, 3],[3, 3]], dtype="float32")
            >>> y = paddle.to_tensor([[3, 3],[3, 1]], dtype="float32")
            >>> out = paddle.dist(x, y, 0)
            >>> print(out)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.)

            >>> out = paddle.dist(x, y, 2)
            >>> print(out)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            2.)

            >>> out = paddle.dist(x, y, float("inf"))
            >>> print(out)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            2.)

            >>> out = paddle.dist(x, y, float("-inf"))
            >>> print(out)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.)
    """
    if in_dynamic_mode():
        return _C_ops.dist(x, y, p)

    check_variable_and_dtype(
        x, 'dtype', ['bfloat16', 'float16', 'float32', 'float64'], 'dist'
    )
    check_variable_and_dtype(
        y, 'dtype', ['bfloat16', 'float16', 'float32', 'float64'], 'dist'
    )
    check_type(p, 'p', (float, int), 'dist')
    helper = LayerHelper("dist", **locals())
    out = helper.create_variable_for_type_inference(x.dtype)

    inputs = {"X": [x], "Y": [y]}
    outputs = {'Out': [out]}
    attrs = {"p": float(p)}
    helper.append_op(
        type='dist', inputs=inputs, outputs={'Out': out}, attrs=attrs
    )
    return out


def cond(x, p=None, name=None):
    """

    Computes the condition number of a matrix or batches of matrices with respect to a matrix norm ``p``.

    Args:
        x (Tensor): The input tensor could be tensor of shape ``(*, m, n)`` where ``*`` is zero or more batch dimensions
            for ``p`` in ``(2, -2)``, or of shape ``(*, n, n)`` where every matrix is invertible for any supported ``p``.
            And the input data type could be ``float32`` or ``float64``.
        p (float|string, optional): Order of the norm. Supported values are `fro`, `nuc`, `1`, `-1`, `2`, `-2`,
            `inf`, `-inf`. Default value is `None`, meaning that the order of the norm is `2`.
        name (str, optional): The default value is `None`. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: computing results of condition number, its data type is the same as input Tensor ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> x = paddle.to_tensor([[1., 0, -1], [0, 1, 0], [1, 0, 1]])

            >>> # compute conditional number when p is None
            >>> out = paddle.linalg.cond(x)
            >>> print(out)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.41421378)

            >>> # compute conditional number when order of the norm is 'fro'
            >>> out_fro = paddle.linalg.cond(x, p='fro')
            >>> print(out_fro)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            3.16227770)

            >>> # compute conditional number when order of the norm is 'nuc'
            >>> out_nuc = paddle.linalg.cond(x, p='nuc')
            >>> print(out_nuc)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            9.24264145)

            >>> # compute conditional number when order of the norm is 1
            >>> out_1 = paddle.linalg.cond(x, p=1)
            >>> print(out_1)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            2.)

            >>> # compute conditional number when order of the norm is -1
            >>> out_minus_1 = paddle.linalg.cond(x, p=-1)
            >>> print(out_minus_1)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.)

            >>> # compute conditional number when order of the norm is 2
            >>> out_2 = paddle.linalg.cond(x, p=2)
            >>> print(out_2)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.41421378)

            >>> # compute conditional number when order of the norm is -1
            >>> out_minus_2 = paddle.linalg.cond(x, p=-2)
            >>> print(out_minus_2)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.70710671)

            >>> # compute conditional number when order of the norm is inf
            >>> out_inf = paddle.linalg.cond(x, p=float("inf"))
            >>> print(out_inf)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            2.)

            >>> # compute conditional number when order of the norm is -inf
            >>> out_minus_inf = paddle.linalg.cond(x, p=-float("inf"))
            >>> print(out_minus_inf)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.)

            >>> a = paddle.randn([2, 4, 4])
            >>> print(a)
            Tensor(shape=[2, 4, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[ 0.06132207,  1.11349595,  0.41906244, -0.24858207],
              [-1.85169315, -1.50370061,  1.73954511,  0.13331604],
              [ 1.66359663, -0.55764782, -0.59911072, -0.57773495],
              [-1.03176904, -0.33741450, -0.29695082, -1.50258386]],
             [[ 0.67233968, -1.07747352,  0.80170447, -0.06695852],
              [-1.85003340, -0.23008066,  0.65083790,  0.75387722],
              [ 0.61212337, -0.52664012,  0.19209868, -0.18707706],
              [-0.00711021,  0.35236868, -0.40404350,  1.28656745]]])

            >>> a_cond_fro = paddle.linalg.cond(a, p='fro')
            >>> print(a_cond_fro)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [6.37173700 , 35.15114594])

            >>> b = paddle.randn([2, 3, 4])
            >>> print(b)
            Tensor(shape=[2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[ 0.03306439,  0.70149767,  0.77064633, -0.55978841],
              [-0.84461296,  0.99335045, -1.23486686,  0.59551388],
              [-0.63035583, -0.98797107,  0.09410731,  0.47007179]],
             [[ 0.85850012, -0.98949534, -1.63086998,  1.07340240],
              [-0.05492965,  1.04750168, -2.33754158,  1.16518629],
              [ 0.66847134, -1.05326962, -0.05703246, -0.48190674]]])

            >>> b_cond_2 = paddle.linalg.cond(b, p=2)
            >>> print(b_cond_2)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [2.86566353, 6.85834455])

    """

    def mat_norm(input, porder=1.0, axis=None):
        """
        NOTE:
            Calculate the matrix norm of a square matrix or batches of square matrices,
            when porder is in (1, -1, inf, -inf)
        """
        if in_dynamic_mode():
            abs_out = _C_ops.abs(input)
            sum_out = _C_ops.sum(abs_out, axis, None, False)

            if porder == 1 or porder == np.inf:
                return _C_ops.max(sum_out, [-1], False)
            if porder == -1 or porder == -np.inf:
                return _C_ops.min(sum_out, [-1], False)
        else:
            block = LayerHelper('norm', **locals())
            abs_out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            sum_out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            block.append_op(
                type='abs', inputs={'X': input}, outputs={'Out': abs_out}
            )

            reduce_all, axis = _get_reduce_axis(axis, x)
            block.append_op(
                type='reduce_sum',
                inputs={'X': abs_out},
                outputs={'Out': sum_out},
                attrs={
                    'dim': axis,
                    'keep_dim': False,
                    'reduce_all': reduce_all,
                },
            )
            if porder == 1 or porder == np.inf:
                block.append_op(
                    type='reduce_max',
                    inputs={'X': sum_out},
                    outputs={'Out': out},
                    attrs={
                        'dim': [-1],
                        'keep_dim': False,
                        'reduce_all': reduce_all,
                    },
                )
            if porder == -1 or porder == -np.inf:
                block.append_op(
                    type='reduce_min',
                    inputs={'X': sum_out},
                    outputs={'Out': out},
                    attrs={
                        'dim': [-1],
                        'keep_dim': False,
                        'reduce_all': reduce_all,
                    },
                )
            return out

    def fro_norm(input, porder=2, axis=[-1]):
        """
        NOTE:
            Calculate the frobenius norm of a square matrix or batches of square matrices.
        """
        if in_dynamic_mode():
            pow_out = _C_ops.pow(input, porder)
            sum_out_1 = _C_ops.sum(pow_out, axis, None, False)
            sum_out_2 = _C_ops.sum(sum_out_1, axis, None, False)
            return _C_ops.pow(sum_out_2, float(1.0 / porder))
        else:
            block = LayerHelper('norm', **locals())
            pow_out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            sum_out_1 = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            sum_out_2 = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            block.append_op(
                type='pow',
                inputs={'X': input},
                outputs={'Out': pow_out},
                attrs={'factor': porder},
            )

            reduce_all, axis = _get_reduce_axis(axis, x)
            block.append_op(
                type='reduce_sum',
                inputs={'X': pow_out},
                outputs={'Out': sum_out_1},
                attrs={
                    'dim': axis,
                    'keep_dim': False,
                    'reduce_all': reduce_all,
                },
            )
            block.append_op(
                type='reduce_sum',
                inputs={'X': sum_out_1},
                outputs={'Out': sum_out_2},
                attrs={
                    'dim': axis,
                    'keep_dim': False,
                    'reduce_all': reduce_all,
                },
            )
            block.append_op(
                type='pow',
                inputs={'X': sum_out_2},
                outputs={'Out': out},
                attrs={'factor': float(1.0 / porder)},
            )
            return out

    def svd_norm(input, porder, axis=[-1]):
        """
        NOTE:
            Calculate the matrix norm, which is related to singular values, of a matrix
            or batches of matrices, including nuclear norm, 2-norm and (-2)-norm.
        """
        u, s, vh = svd(input, full_matrices=False)

        if in_dynamic_mode():
            if porder == "nuc":
                return _C_ops.sum(s, axis, None, False)
            max_out = _C_ops.max(s, axis, False)
            min_out = _C_ops.min(s, axis, False)
            if porder == 2:
                return _C_ops.divide(max_out, min_out)
            if porder == -2:
                return _C_ops.divide(min_out, max_out)
        else:
            reduce_all, axis = _get_reduce_axis(axis, x)
            block = LayerHelper('norm', **locals())
            out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            if porder == "nuc":
                block.append_op(
                    type='reduce_sum',
                    inputs={'X': s},
                    outputs={'Out': out},
                    attrs={
                        'dim': axis,
                        'keep_dim': False,
                        'reduce_all': reduce_all,
                    },
                )
                return out
            max_out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            min_out = block.create_variable_for_type_inference(
                dtype=block.input_dtype()
            )
            block.append_op(
                type='reduce_max',
                inputs={'X': s},
                outputs={'Out': max_out},
                attrs={
                    'dim': axis,
                    'keep_dim': False,
                    'reduce_all': reduce_all,
                },
            )
            block.append_op(
                type='reduce_min',
                inputs={'X': s},
                outputs={'Out': min_out},
                attrs={
                    'dim': axis,
                    'keep_dim': False,
                    'reduce_all': reduce_all,
                },
            )
            if porder == 2:
                block.append_op(
                    type='elementwise_div',
                    inputs={'X': max_out, 'Y': min_out},
                    outputs={'Out': out},
                    attrs={'aixs': axis, 'use_mkldnn': False},
                )
                return out
            if porder == -2:
                block.append_op(
                    type='elementwise_div',
                    inputs={'X': min_out, 'Y': max_out},
                    outputs={'Out': out},
                    attrs={'aixs': axis, 'use_mkldnn': False},
                )
                return out

    def empty_tensor(input, shape):
        if in_dynamic_mode():
            return input.reshape(shape)
        raise ValueError(
            "only support x is nonempty tensor in static graph mode"
        )

    x_shape = list(x.shape)
    if not len(x_shape) >= 2:
        raise ValueError(
            "input should be a matrix or batches of matrices, "
            + f"but the dimention of received input is {len(x_shape)}"
        )
    if p is None:
        p = 2
    x_size = 0 if (0 in x_shape) else 1
    if p in ("fro", "nuc", 1, -1, np.inf, -np.inf):
        if x_shape[len(x_shape) - 1] == x_shape[len(x_shape) - 2]:
            if x_size == 0:
                return empty_tensor(x, x_shape[:-2])
            x_inv = x.inverse()
            if p == "fro":
                return fro_norm(x) * fro_norm(x_inv)
            if p == "nuc":
                return svd_norm(x, p) * svd_norm(x_inv, p)
            if p in (1, -1):
                return mat_norm(x, porder=p, axis=[-2]) * mat_norm(
                    x_inv, porder=p, axis=[-2]
                )
            if p in (np.inf, -np.inf):
                return mat_norm(x, porder=p, axis=[-1]) * mat_norm(
                    x_inv, porder=p, axis=[-1]
                )
        else:
            raise ValueError(
                f"only support p is {p} when input is a "
                + "square matrix or batches of square matrices"
            )
    elif p in (2, -2):
        if x_size == 0:
            return empty_tensor(x, x_shape[:-2])
        return svd_norm(x, porder=p)
    else:
        raise ValueError(
            f"unsupported {p} for p, only supporting ('fro', 'nuc', "
            + "1, -1, 2, -2, inf, -inf) or none"
        )


def dot(x, y, name=None):
    """
    This operator calculates inner product for vectors.

    Note:
       Support 1-d and 2-d Tensor. When it is 2d, the first dimension of this matrix
       is the batch dimension, which means that the vectors of multiple batches are dotted.

    Parameters:
        x(Tensor): 1-D or 2-D ``Tensor``. Its dtype should be ``float32``, ``float64``, ``int32``, ``int64``, ``complex64``, ``complex128``
        y(Tensor): 1-D or 2-D ``Tensor``. Its dtype should be ``float32``, ``float64``, ``int32``, ``int64``, ``complex64``, ``complex128``
        name(str, optional): Name of the output. Default is None. It's used to print debug info for developers. Details: :ref:`api_guide_Name`

    Returns:
        Tensor: the calculated result Tensor.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> # 1-D Tensor * 1-D Tensor
            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.to_tensor([4, 5, 6])
            >>> z = paddle.dot(x, y)
            >>> print(z)
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
            32)

            >>> # 2-D Tensor * 2-D Tensor
            >>> x = paddle.to_tensor([[1, 2, 3], [2, 4, 6]])
            >>> y = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
            >>> z = paddle.dot(x, y)
            >>> print(z)
            Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [32, 64])

    """
    if in_dynamic_mode():
        return _C_ops.dot(x, y)
    else:
        op_type = 'dot'

        assert x is not None, f'x cannot be None in {op_type}'
        assert y is not None, f'y cannot be None in {op_type}'

        check_variable_and_dtype(
            x,
            'x',
            [
                'float16',
                'uint16',
                'float32',
                'float64',
                'int32',
                'int64',
                'complex64',
                'complex128',
            ],
            op_type,
        )
        check_variable_and_dtype(
            y,
            'y',
            [
                'float16',
                'uint16',
                'float32',
                'float64',
                'int32',
                'int64',
                'complex64',
                'complex128',
            ],
            op_type,
        )

        helper = LayerHelper(op_type, **locals())
        if name is None:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
        else:
            out = helper.create_variable(
                name=name, dtype=x.dtype, persistable=False
            )
        helper.append_op(
            type="dot", inputs={'X': x, 'Y': y}, attrs={}, outputs={"Out": out}
        )
        return out


def cov(x, rowvar=True, ddof=True, fweights=None, aweights=None, name=None):
    """
    Estimate the covariance matrix of the input variables, given data and weights.

    A covariance matrix is a square matrix, indicate the covariance of each pair variables in the input matrix.
    For example, for an N-dimensional samples X=[x1,x2,…xN]T, then the covariance matrix
    element Cij is the covariance of xi and xj. The element Cii is the variance of xi itself.

    Parameters:
        x (Tensor): A N-D(N<=2) Tensor containing multiple variables and observations. By default, each row of x represents a variable. Also see rowvar below.
        rowvar (Bool, optional): If rowvar is True (default), then each row represents a variable, with observations in the columns. Default: True.
        ddof (Bool, optional): If ddof=True will return the unbiased estimate, and ddof=False will return the simple average. Default: True.
        fweights (Tensor, optional): 1-D Tensor of integer frequency weights; The number of times each observation vector should be repeated. Default: None.
        aweights (Tensor, optional): 1-D Tensor of observation vector weights. How important of the observation vector, larger data means this element is more important. Default: None.
        name (str, optional): Name of the output. Default is None. It's used to print debug info for developers. Details: :ref:`api_guide_Name` .

    Returns:
        Tensor: The covariance matrix Tensor of the variables.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> xt = paddle.rand((3, 4))
            >>> paddle.linalg.cov(xt)
            >>> print(xt)
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.86583614, 0.52014720, 0.25960937, 0.90525323],
             [0.42400089, 0.40641287, 0.97020894, 0.74437362],
             [0.51785129, 0.73292869, 0.97786582, 0.04315904]])
    """
    op_type = 'cov'
    if len(x.shape) > 2 or len(x.shape) < 1:
        raise ValueError(
            "Input(x) only support N-D (1<=N<=2) tensor in cov, but received "
            "length of Input(input) is %s." % len(x.shape)
        )
    check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'cov')
    nx = x
    if len(x.shape) == 1:
        nx = x.reshape((1, -1))
    if not rowvar and nx.shape[0] != 1:
        nx = nx.t()
    w = None
    observation_num = nx.shape[1]
    if fweights is not None:
        w = fweights.astype(nx.dtype)
        if len(w.shape) > 1:
            raise ValueError(
                "Input(fweights) only support N-D (N<=1) tensor in cov, but received "
                "shape of Input(input) is %s." % len(fweights.shape)
            )
        if fweights.shape[0] != observation_num:
            raise ValueError(
                "The number of Input(fweights) should equal to x's dim[1]: {}, but received "
                "size of Input(fweights) is {}.".format(
                    observation_num, fweights.shape[0]
                )
            )
        if fweights.min() < 0:
            raise ValueError(
                "The value of Input(fweights) cannot be negtive, but received "
                f"min of Input(fweights) is {fweights.min()}."
            )
        if not paddle.all(fweights == paddle.round(fweights.astype('float64'))):
            raise ValueError("Input(fweights) must be integer ")

    if aweights is not None:
        aw = aweights.astype(nx.dtype)
        if len(aw.shape) > 1:
            raise ValueError(
                "Input(aweights) only support N-D (N<=1) tensor in cov, but received "
                "length of Input(input) is %s." % len(aweights.shape)
            )
        check_variable_and_dtype(
            aweights, 'dtype', ['float32', 'float64'], 'cov'
        )
        if aweights.shape[0] != observation_num:
            raise ValueError(
                "The number of Input(aweights) should equal to x's dim[1]: {}, but received "
                "size of Input(aweights) is {}.".format(
                    observation_num, aweights.shape[0]
                )
            )
        if aweights.min() < 0:
            raise ValueError(
                "The value of Input(aweights) cannot be negtive, but received "
                f"min of Input(aweights) is {aweights.min()}."
            )
        if w is not None:
            w = w * aw
        else:
            w = aw

    w_sum = paddle.to_tensor(observation_num, dtype=nx.dtype)
    if fweights is not None or aweights is not None:
        w_sum = w.sum()
        if w_sum.item() == 0:
            raise ValueError("The sum of weights is zero, can't be normalized.")

    if w is not None:
        nx_w = nx * w
        avg = (nx_w).sum(axis=1) / w_sum
    else:
        avg = nx.sum(axis=1) / w_sum
        nx_w = nx

    if w is not None and aweights is not None and ddof:
        norm_factor = w_sum - (w * aweights).sum() / w_sum
    else:
        norm_factor = w_sum - ddof
    if norm_factor <= 0:
        norm_factor = paddle.to_tensor(0, dtype=nx.dtype)
    nx = nx - avg.unsqueeze(1)
    xxt = paddle.mm(nx, nx_w.t().conj())
    cov = paddle.divide(xxt, norm_factor).squeeze()
    return cov


def t(input, name=None):
    """
    Transpose <=2-D tensor.
    0-D and 1-D tensors are returned as it is and 2-D tensor is equal to
    the paddle.transpose function which perm dimensions set 0 and 1.

    Args:
        input (Tensor): The input Tensor. It is a N-D (N<=2) Tensor of data types float32, float64, int32, int64.
        name (str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor: A transposed n-D Tensor, with data type being float16, float32, float64, int32, int64.

    Examples:

        .. code-block:: python
            :name: code-example

            >>> import paddle

            >>> # Example 1 (0-D tensor)
            >>> x = paddle.to_tensor([0.79])
            >>> out = paddle.t(x)
            >>> print(out)
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.79000002])

            >>> # Example 2 (1-D tensor)
            >>> x = paddle.to_tensor([0.79, 0.84, 0.32])
            >>> out2 = paddle.t(x)
            >>> print(out2)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.79000002, 0.83999997, 0.31999999])
            >>> print(paddle.t(x).shape)
            [3]

            >>> # Example 3 (2-D tensor)
            >>> x = paddle.to_tensor([[0.79, 0.84, 0.32],
            ...                       [0.64, 0.14, 0.57]])
            >>> print(x.shape)
            [2, 3]
            >>> out3 = paddle.t(x)
            >>> print(out3)
            Tensor(shape=[3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.79000002, 0.63999999],
             [0.83999997, 0.14000000],
             [0.31999999, 0.56999999]])
            >>> print(paddle.t(x).shape)
            [3, 2]

    """
    if len(input.shape) > 2:
        raise ValueError(
            "Input(input) only support N-D (N<=2) tensor, but received "
            "length of Input(input) is %s. Perhaps you can use paddle."
            "tensor.transpose() instead." % len(input.shape)
        )
    if in_dynamic_mode():
        if len(input.shape) <= 1:
            return input
        # 2-D tensor
        perm = [1, 0]
        out = _C_ops.transpose(input, perm)
        return out
    else:
        check_variable_and_dtype(
            input,
            'input',
            ['float16', 'float32', 'float64', 'int32', 'int64'],
            'transpose',
        )

        helper = LayerHelper('t', **locals())
        out = helper.create_variable_for_type_inference(input.dtype)
        input_shape = helper.create_variable_for_type_inference(input.dtype)
        if len(input.shape) <= 1:
            out = input
        else:
            helper.append_op(
                type='transpose2',
                inputs={'X': [input]},
                outputs={'Out': [out], 'XShape': [input_shape]},
                attrs={'axis': [1, 0]},
            )
        return out


@inplace_apis_in_dygraph_only
def t_(input, name=None):
    r"""
    Inplace version of ``t`` API, the output Tensor will be inplaced with input ``input``.
    Please refer to :ref:`api_paddle_t`.
    """
    if len(input.shape) > 2:
        raise ValueError(
            "Input(input) only support N-D (N<=2) tensor, but received "
            "length of Input(input) is %s. Perhaps you can use paddle."
            "tensor.transpose() instead." % len(input.shape)
        )
    if in_dynamic_mode():
        if len(input.shape) <= 1:
            return input
        # 2-D tensor
        perm = [1, 0]
        out = _C_ops.transpose_(input, perm)
        return out


def cross(x, y, axis=9, name=None):
    """
    Computes the cross product between two tensors along an axis.

    Inputs must have the same shape, and the length of their axes should be equal to 3.
    If `axis` is not given, it defaults to the first axis found with the length 3.

    Args:
        x (Tensor): The first input tensor, the data type is float16, float32, float64, int32, int64.
        y (Tensor): The second input tensor, the data type is float16, float32, float64, int32, int64.
        axis (int, optional): The axis along which to compute the cross product. It defaults to be 9 which indicates using the first axis found with the length 3.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. A Tensor with same data type as `x`.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.0, 1.0, 1.0],
            ...                         [2.0, 2.0, 2.0],
            ...                         [3.0, 3.0, 3.0]])
            >>> y = paddle.to_tensor([[1.0, 1.0, 1.0],
            ...                         [1.0, 1.0, 1.0],
            ...                         [1.0, 1.0, 1.0]])
            ...
            >>> z1 = paddle.cross(x, y)
            >>> print(z1)
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-1., -1., -1.],
             [ 2.,  2.,  2.],
             [-1., -1., -1.]])

            >>> z2 = paddle.cross(x, y, axis=1)
            >>> print(z2)
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]])
    """
    if in_dynamic_mode():
        axis = K_DEFAULT_DIM if axis is None else axis
        return _C_ops.cross(x, y, axis)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'uint16', 'float32', 'float64', "int32", "int64"],
            'cross',
        )
        check_variable_and_dtype(
            y,
            'y',
            ['float16', 'uint16', 'float32', 'float64', "int32", "int64"],
            'cross',
        )
        helper = LayerHelper("cross", **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        attrs = {}
        attrs['dim'] = axis

        helper.append_op(
            type='cross',
            inputs={'X': x, 'Y': y},
            outputs={'Out': out},
            attrs=attrs,
        )
        return out


def cholesky(x, upper=False, name=None):
    r"""
    Computes the Cholesky decomposition of one symmetric positive-definite
    matrix or batches of symmetric positive-definite matrice.

    If `upper` is `True`, the decomposition has the form :math:`A = U^{T}U` ,
    and the returned matrix :math:`U` is upper-triangular. Otherwise, the
    decomposition has the form  :math:`A = LL^{T}` , and the returned matrix
    :math:`L` is lower-triangular.

    Args:
        x (Tensor): The input tensor. Its shape should be `[*, M, M]`,
            where * is zero or more batch dimensions, and matrices on the
            inner-most 2 dimensions all should be symmetric positive-definite.
            Its data type should be float32 or float64.
        upper (bool, optional): The flag indicating whether to return upper or lower
            triangular matrices. Default: False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, A Tensor with same shape and data type as `x`. It represents
        triangular matrices generated by Cholesky decomposition.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> a = paddle.rand([3, 3], dtype="float32")
            >>> a_t = paddle.transpose(a, [1, 0])
            >>> x = paddle.matmul(a, a_t) + 1e-03

            >>> out = paddle.linalg.cholesky(x, upper=False)
            >>> print(out)
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1.04337072, 0.        , 0.        ],
             [1.06467664, 0.17859250, 0.        ],
             [1.30602181, 0.08326444, 0.22790681]])
    """
    if in_dynamic_mode():
        return _C_ops.cholesky(x, upper)
    else:
        check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'cholesky')
        check_type(upper, 'upper', bool, 'cholesky')
        helper = LayerHelper('cholesky', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='cholesky',
            inputs={'X': [x]},
            outputs={'Out': out},
            attrs={'upper': upper},
        )
        return out


def matrix_rank(x, tol=None, hermitian=False, name=None):
    r"""
    Computes the rank of a matrix.

    The rank of a matrix is the number of singular values that are greater than the specified `tol` threshold when hermitian=False,
    or the number of eigenvalues in absolute value that are greater than the specified `tol` threshold when hermitian=True.

    Args:
        x (Tensor): The input tensor. Its shape should be `[..., m, n]`, where `...` is zero or more batch dimensions. If `x` is a batch
            of matrices then the output has the same batch dimensions. The data type of `x` should be float32 or float64.
        tol (float|Tensor, optional): the tolerance value. If `tol` is not specified, and `sigma` is the largest singular value
            (or eigenvalues in absolute value), and `eps` is the epsilon value for the dtype of `x`, then `tol` is computed with formula
            `tol=sigma * max(m,n) * eps`. Note that if `x` is a batch of matrices, `tol` is computed this way for every batch. Default: None.
        hermitian (bool, optional): indicates whether `x` is Hermitian. Default: False. When hermitian=True, `x` is assumed to be Hermitian,
            enabling a more efficient method for finding eigenvalues, but `x` is not checked inside the function. Instead, We just use
            the lower triangular of the matrix to compute. Default: False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Rank of tensor x.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> a = paddle.eye(10)
            >>> b = paddle.linalg.matrix_rank(a)
            >>> print(b)
            Tensor(shape=[], dtype=int32, place=Place(cpu), stop_gradient=True,
            10)

            >>> c = paddle.ones(shape=[3, 4, 5, 5])
            >>> d = paddle.linalg.matrix_rank(c, tol=0.01, hermitian=True)
            >>> print(d)
            Tensor(shape=[3, 4], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[1, 1, 1, 1],
             [1, 1, 1, 1],
             [1, 1, 1, 1]])

    """
    if in_dynamic_mode():
        if isinstance(tol, Variable):
            if tol.dtype != x.dtype:
                tol_tensor = cast(tol, x.dtype)
            else:
                tol_tensor = tol
            use_default_tol = False
            return _C_ops.matrix_rank_tol(
                x, tol_tensor, use_default_tol, hermitian
            )

        if tol is None:
            tol_attr = 0.0
            use_default_tol = True
        else:
            tol_attr = float(tol)
            use_default_tol = False
        return _C_ops.matrix_rank(x, tol_attr, use_default_tol, hermitian)
    else:
        inputs = {}
        attrs = {}
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'matrix_rank')
        inputs['X'] = x
        if tol is None:
            attrs['use_default_tol'] = True
        elif isinstance(tol, Variable):
            attrs['use_default_tol'] = False
            if tol.dtype != x.dtype:
                inputs['TolTensor'] = cast(tol, x.dtype)
            else:
                inputs['TolTensor'] = tol
        else:
            check_type(tol, 'tol', float, 'matrix_rank')
            attrs['use_default_tol'] = False
            attrs['tol'] = tol
        check_type(hermitian, 'hermitian', bool, 'matrix_rank')
        attrs['hermitian'] = hermitian

        helper = LayerHelper('matrix_rank', **locals())
        out = helper.create_variable_for_type_inference(dtype='int32')
        helper.append_op(
            type='matrix_rank', inputs=inputs, outputs={'Out': out}, attrs=attrs
        )
        return out


def bmm(x, y, name=None):
    """
    Applies batched matrix multiplication to two tensors.

    Both of the two input tensors must be three-dementional and share the same batch size.

    If x is a (b, m, k) tensor, y is a (b, k, n) tensor, the output will be a (b, m, n) tensor.

    Args:
        x (Tensor): The input Tensor.
        y (Tensor): The input Tensor.
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically. Default: None.

    Returns:
        Tensor: The product Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # In imperative mode:
            >>> # size x: (2, 2, 3) and y: (2, 3, 2)
            >>> x = paddle.to_tensor([[[1.0, 1.0, 1.0],
            ...                     [2.0, 2.0, 2.0]],
            ...                     [[3.0, 3.0, 3.0],
            ...                     [4.0, 4.0, 4.0]]])
            >>> y = paddle.to_tensor([[[1.0, 1.0],[2.0, 2.0],[3.0, 3.0]],
            ...                     [[4.0, 4.0],[5.0, 5.0],[6.0, 6.0]]])
            >>> out = paddle.bmm(x, y)
            >>> print(out)
            Tensor(shape=[2, 2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[6. , 6. ],
              [12., 12.]],
             [[45., 45.],
              [60., 60.]]])

    """
    if in_dynamic_mode():
        return _C_ops.bmm(x, y)
    else:
        x_shape = x.shape
        y_shape = y.shape
        if not len(x_shape) == len(y_shape) == 3:
            raise ValueError(
                "x and y should be 3-dimensional. But received x's dimention: {}, y's dimention: {}".format(
                    x_shape, y_shape
                )
            )
        if x_shape[2] != -1 and y_shape[1] != -1 and x_shape[2] != y_shape[1]:
            raise ValueError(
                "x's width must be equal with y's height. But received x's shape: {}, y's shape: {}".format(
                    x_shape, y_shape
                )
            )
        if x_shape[0] != -1 and y_shape[0] != -1 and x_shape[0] != y_shape[0]:
            raise ValueError(
                "x's batch (shape[0]) must be equal with y's batch (shape[0]). But received x's shape: {}, y's shape: {}".format(
                    x_shape, y_shape
                )
            )
        helper = LayerHelper('bmm', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='bmm', inputs={'X': x, 'Y': y}, outputs={'Out': out}
        )
        return out


def histogram(input, bins=100, min=0, max=0, name=None):
    """
    Computes the histogram of a tensor. The elements are sorted into equal width bins between min and max.
    If min and max are both zero, the minimum and maximum values of the data are used.

    Args:
        input (Tensor): A Tensor(or LoDTensor) with shape :math:`[N_1, N_2,..., N_k]` . The data type of the input Tensor
            should be float32, float64, int32, int64.
        bins (int, optional): number of histogram bins. Default: 100.
        min (int, optional): lower end of the range (inclusive). Default: 0.
        max (int, optional): upper end of the range (inclusive). Default: 0.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: data type is int64, shape is (nbins,).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> inputs = paddle.to_tensor([1, 2, 1])
            >>> result = paddle.histogram(inputs, bins=4, min=0, max=3)
            >>> print(result)
            Tensor(shape=[4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 2, 1, 0])
    """
    if in_dynamic_mode():
        return _C_ops.histogram(input, bins, min, max)
    else:
        helper = LayerHelper('histogram', **locals())
        check_variable_and_dtype(
            input, 'X', ['int32', 'int64', 'float32', 'float64'], 'histogram'
        )
        out = helper.create_variable_for_type_inference(VarDesc.VarType.INT64)
        helper.append_op(
            type='histogram',
            inputs={'X': input},
            outputs={'Out': out},
            attrs={'bins': bins, 'min': min, 'max': max},
        )
        return out


def bincount(x, weights=None, minlength=0, name=None):
    """
    Computes frequency of each value in the input tensor.

    Args:
        x (Tensor): A Tensor with non-negative integer. Should be 1-D tensor.
        weights (Tensor, optional): Weight for each value in the input tensor. Should have the same shape as input. Default is None.
        minlength (int, optional): Minimum number of bins. Should be non-negative integer. Default is 0.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        Tensor: The tensor of frequency.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1, 2, 1, 4, 5])
            >>> result1 = paddle.bincount(x)
            >>> print(result1)
            Tensor(shape=[6], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 2, 1, 0, 1, 1])

            >>> w = paddle.to_tensor([2.1, 0.4, 0.1, 0.5, 0.5])
            >>> result2 = paddle.bincount(x, weights=w)
            >>> print(result2)
            Tensor(shape=[6], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 2.19999981, 0.40000001, 0.        , 0.50000000, 0.50000000])
    """
    if x.dtype not in [paddle.int32, paddle.int64]:
        raise TypeError("Elements in Input(x) should all be integers")

    if in_dynamic_mode():
        return _C_ops.bincount(x, weights, minlength)
    else:
        helper = LayerHelper('bincount', **locals())

        check_variable_and_dtype(x, 'X', ['int32', 'int64'], 'bincount')

        if weights is not None:
            check_variable_and_dtype(
                weights,
                'Weights',
                ['int32', 'int64', 'float32', 'float64'],
                'bincount',
            )
            out = helper.create_variable_for_type_inference(dtype=weights.dtype)
        else:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='bincount',
            inputs={'X': x, 'Weights': weights},
            outputs={'Out': out},
            attrs={'minlength': minlength},
        )
        return out


def mv(x, vec, name=None):
    """
    Performs a matrix-vector product of the matrix x and the vector vec.

    Args:
        x (Tensor): A tensor with shape :math:`[M, N]` , The data type of the input Tensor x
            should be one of float32, float64.
        vec (Tensor): A tensor with shape :math:`[N]` , The data type of the input Tensor x
            should be one of float32, float64.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        Tensor: The tensor which is producted by x and vec.

    Examples:
        .. code-block:: python

            >>> # x: [M, N], vec: [N]
            >>> # paddle.mv(x, vec)  # out: [M]

            >>> import paddle

            >>> x = paddle.to_tensor([[2, 1, 3], [3, 0, 1]]).astype("float64")
            >>> vec = paddle.to_tensor([3, 5, 1]).astype("float64")
            >>> out = paddle.mv(x, vec)
            >>> print(out)
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [14., 10.])
    """
    if in_dynamic_mode():
        return _C_ops.mv(x, vec)
    else:

        def __check_input(x, vec):
            var_names = {'x': x, 'vec': vec}
            for name, val in var_names.items():
                check_variable_and_dtype(
                    val, name, ['float32', 'float64'], 'mv'
                )
            x_shape = list(x.shape)
            vec_shape = list(vec.shape)
            if len(x_shape) != 2:
                raise ValueError(
                    f"x should be 2-dimensional. But received x's dimention: {x_shape}"
                )
            if len(vec_shape) != 1:
                raise ValueError(
                    "vec should be 1-dimensional. But received vec's dimention: {}".format(
                        vec_shape
                    )
                )

        __check_input(x, vec)

        helper = LayerHelper('mv', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='mv', inputs={'X': x, 'Vec': vec}, outputs={'Out': out}
        )
        return out


def det(x, name=None):
    """

    Calculates determinant value of a square matrix or batches of square matrices.

    Args:
        x (Tensor): the input matrix of size `(n, n)` or the
            batch of matrices of size `(*, n, n)` where `*` is one or more
            batch dimensions.
        name (str, optional): Name of the output.It's used to print debug info for
            developers. Details: :ref:`api_guide_Name`. Default is None.

    Returns:
        Tensor, the determinant value of a square matrix or batches of square matrices.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> x =  paddle.randn([3,3,3])
            >>> A = paddle.linalg.det(x)
            >>> print(A)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-1.29280925,  0.77832544,  0.89754158])


    """
    if in_dynamic_mode():
        return _C_ops.det(x)
    else:
        check_dtype(x.dtype, 'Input', ['float16', 'float32', 'float64'], 'det')

        input_shape = list(x.shape)
        assert len(input_shape) >= 2, (
            "The x must be at least 2-dimensional, "
            "but received Input x's dimensional: %s.\n" % len(input_shape)
        )

        assert (
            input_shape[-1] == input_shape[-2]
        ), "Expect squared input," "but received {} by {} matrix.\n".format(
            input_shape[-2],
            input_shape[-1],
        )
        helper = LayerHelper('determinant', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='determinant', inputs={'Input': [x]}, outputs={'Out': [out]}
        )
        return out


def slogdet(x, name=None):
    """

    Calculates the sign and natural logarithm of the absolute value of a square matrix's or batches square matrices' determinant.
    The determinant can be computed with ``sign * exp`` (logabsdet).

    Supports input of float, double.

    Note that for matrices that have zero determinant, this returns ``(0, -inf)``.

    Args:
        x (Tensor): the batch of matrices of size :math:`(*, n, n)`
            where math:`*` is one or more batch dimensions.
        name (str, optional): Name of the output.It's used to print debug info for
            developers. Details: :ref:`api_guide_Name`. Default is None.

    Returns:
        y (Tensor), A tensor containing the sign of the determinant and the natural logarithm
        of the absolute value of determinant, respectively.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> x =  paddle.randn([3,3,3])
            >>> A = paddle.linalg.slogdet(x)
            >>> print(A)
            >>> # doctest: +SKIP
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-1.        ,  1.        ,  1.        ],
             [ 0.25681755, -0.25061053, -0.10809582]])
            >>> # doctest: -SKIP

    """
    if in_dynamic_mode():
        return _C_ops.slogdet(x)
    else:
        check_dtype(x.dtype, 'Input', ['float32', 'float64'], 'slogdet')

        input_shape = list(x.shape)
        assert len(input_shape) >= 2, (
            "The x must be at least 2-dimensional, "
            "but received Input x's dimensional: %s.\n" % len(input_shape)
        )

        assert (
            input_shape[-1] == input_shape[-2]
        ), "Expect squared input," "but received {} by {} matrix.\n".format(
            input_shape[-2],
            input_shape[-1],
        )
        helper = LayerHelper('slogdeterminant', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='slogdeterminant',
            inputs={'Input': [x]},
            outputs={'Out': [out]},
        )
        return out


def svd(x, full_matrices=False, name=None):
    r"""
    Computes the singular value decomposition of one matrix or a batch of regular matrices.

    Let :math:`X` be the input matrix or a batch of input matrices, the output should satisfies:

    .. math::
        X = U * diag(S) * VT

    Args:
        x (Tensor): The input tensor. Its shape should be `[..., N, M]`,
            where `...` is zero or more batch dimensions. N and M can be arbitraty
            positive number. Note that if x is sigular matrices, the grad is numerical
            instable. The data type of x should be float32 or float64.
        full_matrices (bool, optional): A flag to control the behavor of svd.
            If full_matrices = True, svd op will compute full U and V matrics,
            which means shape of U is `[..., N, N]`, shape of V is `[..., M, M]`. K = min(M, N).
            If full_matrices = False, svd op will use a economic method to store U and V.
            which means shape of U is `[..., N, K]`, shape of V is `[..., M, K]`. K = min(M, N).
            Default value is False.
        name (str, optional): Name for the operation. For more information,
            please refer to :ref:`api_guide_Name`. Default value is None.

    Returns:
        - U (Tensor), is the singular value decomposition result U.
        - S (Tensor), is the singular value decomposition result S.
        - VH (Tensor), VH is the conjugate transpose of V, which is the singular value decomposition result V.

        Tuple of 3 tensors(U, S, VH): VH is the conjugate transpose of V. S is the singlar value vectors of matrics with shape `[..., K]`

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.0, 2.0], [1.0, 3.0], [4.0, 6.0]]).astype('float64')
            >>> x = x.reshape([3, 2])
            >>> u, s, vh = paddle.linalg.svd(x)
            >>> print (u)
            Tensor(shape=[3, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[-0.27364809, -0.21695147],
             [-0.37892198, -0.87112408],
             [-0.88404460,  0.44053933]])

            >>> print (s)
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [8.14753743, 0.78589688])

            >>> print (vh)
            Tensor(shape=[2, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[-0.51411221, -0.85772294],
             [ 0.85772294, -0.51411221]])

            >>> # one can verify : U * S * VT == X
            >>> #                  U * UH == I
            >>> #                  V * VH == I
    """

    if in_dynamic_mode():
        return _C_ops.svd(x, full_matrices)
    else:
        check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'svd')
        check_type(full_matrices, 'full_matrices', bool, 'svd')
        helper = LayerHelper('svd', **locals())
        u = helper.create_variable_for_type_inference(dtype=x.dtype)
        vh = helper.create_variable_for_type_inference(dtype=x.dtype)
        s = helper.create_variable_for_type_inference(dtype=x.dtype)
        attrs = {}
        attrs['full_matrices'] = full_matrices
        helper.append_op(
            type='svd',
            inputs={'X': [x]},
            outputs={'U': u, 'VH': vh, 'S': s},
            attrs=attrs,
        )
        return u, s, vh


def pca_lowrank(x, q=None, center=True, niter=2, name=None):
    r"""
    Performs linear Principal Component Analysis (PCA) on a low-rank matrix or batches of such matrices.

    Let :math:`X` be the input matrix or a batch of input matrices, the output should satisfies:

    .. math::
        X = U * diag(S) * V^{T}

    Args:
        x (Tensor): The input tensor. Its shape should be `[..., N, M]`,
            where `...` is zero or more batch dimensions. N and M can be arbitraty
            positive number. The data type of x should be float32 or float64.
        q (int, optional): a slightly overestimated rank of :math:`X`.
            Default value is :math:`q=min(6,N,M)`.
        center (bool, optional): if True, center the input tensor.
            Default value is True.
        niter (int, optional): number of iterations to perform. Default: 2.
        name (str, optional): Name for the operation. For more information,
            please refer to :ref:`api_guide_Name`. Default: None.

    Returns:
        - Tensor U, is N x q matrix.
        - Tensor S, is a vector with length q.
        - Tensor V, is M x q matrix.

        tuple (U, S, V): which is the nearly optimal approximation of a singular value decomposition of a centered matrix :math:`X`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> x = paddle.randn((5, 5), dtype='float64')
            >>> U, S, V = paddle.linalg.pca_lowrank(x)
            >>> print(U)
           Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
           [[ 0.80131563,  0.11962647,  0.27667179, -0.25891214,  0.44721360],
            [-0.12642301,  0.69917551, -0.17899393,  0.51296394,  0.44721360],
            [ 0.08997135, -0.69821706, -0.20059228,  0.51396579,  0.44721360],
            [-0.23871837, -0.02815453, -0.59888153, -0.61932365,  0.44721360],
            [-0.52614559, -0.09243040,  0.70179595, -0.14869394,  0.44721360]])

            >>> print(S)
            Tensor(shape=[5], dtype=float64, place=Place(cpu), stop_gradient=True,
            [2.60101614, 2.40554940, 1.49768346, 0.19064830, 0.00000000])

            >>> print(V)
            Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[ 0.58339481, -0.17143771,  0.00522143,  0.57976310,  0.54231640],
             [ 0.22334335,  0.72963474, -0.30148399, -0.39388750,  0.41438019],
             [ 0.05416913,  0.34666487,  0.93549758,  0.00063507,  0.04162998],
             [-0.39519094,  0.53074980, -0.16687419,  0.71175586, -0.16638919],
             [-0.67131070, -0.19071018,  0.07795789, -0.04615811,  0.71046714]])
    """

    def conjugate(x):
        if x.is_complex():
            return x.conj()
        return x

    def transpose(x):
        shape = x.shape
        perm = list(range(0, len(shape)))
        perm = perm[:-2] + [perm[-1]] + [perm[-2]]
        return paddle.transpose(x, perm)

    def transjugate(x):
        return conjugate(transpose(x))

    def get_approximate_basis(x, q, niter=2, M=None):
        niter = 2 if niter is None else niter
        m, n = x.shape[-2:]
        qr = paddle.linalg.qr

        R = paddle.randn((n, q), dtype=x.dtype)

        A_t = transpose(x)
        A_H = conjugate(A_t)
        if M is None:
            Q = qr(paddle.matmul(x, R))[0]
            for i in range(niter):
                Q = qr(paddle.matmul(A_H, Q))[0]
                Q = qr(paddle.matmul(x, Q))[0]
        else:
            M_H = transjugate(M)
            Q = qr(paddle.matmul(x, R) - paddle.matmul(M, R))[0]
            for i in range(niter):
                Q = qr(paddle.matmul(A_H, Q) - paddle.matmul(M_H, Q))[0]
                Q = qr(paddle.matmul(x, Q) - paddle.matmul(M, Q))[0]

        return Q

    def svd_lowrank(x, q=6, niter=2, M=None):
        q = 6 if q is None else q
        m, n = x.shape[-2:]
        if M is None:
            M_t = None
        else:
            M_t = transpose(M)
        A_t = transpose(x)

        if m < n or n > q:
            Q = get_approximate_basis(A_t, q, niter=niter, M=M_t)
            Q_c = conjugate(Q)
            if M is None:
                B_t = paddle.matmul(x, Q_c)
            else:
                B_t = paddle.matmul(x, Q_c) - paddle.matmul(M, Q_c)
            assert B_t.shape[-2] == m, (B_t.shape, m)
            assert B_t.shape[-1] == q, (B_t.shape, q)
            assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
            U, S, Vh = paddle.linalg.svd(B_t, full_matrices=False)
            V = transjugate(Vh)
            V = Q.matmul(V)
        else:
            Q = get_approximate_basis(x, q, niter=niter, M=M)
            Q_c = conjugate(Q)
            if M is None:
                B = paddle.matmul(A_t, Q_c)
            else:
                B = paddle.matmul(A_t, Q_c) - paddle.matmul(M_t, Q_c)
            B_t = transpose(B)
            assert B_t.shape[-2] == q, (B_t.shape, q)
            assert B_t.shape[-1] == n, (B_t.shape, n)
            assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
            U, S, Vh = paddle.linalg.svd(B_t, full_matrices=False)
            V = transjugate(Vh)
            U = Q.matmul(U)

        return U, S, V

    if not paddle.is_tensor(x):
        raise ValueError(f'Input must be tensor, but got {type(x)}')

    (m, n) = x.shape[-2:]

    if q is None:
        q = min(6, m, n)
    elif not (q >= 0 and q <= min(m, n)):
        raise ValueError(
            f'q(={q}) must be non-negative integer'
            f' and not greater than min(m, n)={min(m, n)}'
        )
    if not (niter >= 0):
        raise ValueError(f'niter(={niter}) must be non-negative integer')

    if not center:
        return svd_lowrank(x, q, niter=niter, M=None)

    C = x.mean(axis=-2, keepdim=True)
    return svd_lowrank(x - C, q, niter=niter, M=None)


def matrix_power(x, n, name=None):
    r"""

    Computes the n-th power of a square matrix or a batch of square matrices.

    Let :math:`X` be a sqaure matrix or a batch of square matrices, :math:`n` be
    an exponent, the equation should be:

    .. math::
        Out = X ^ {n}

    Specifically,

    - If `n > 0`, it returns the matrix or a batch of matrices raised to the power of `n`.

    - If `n = 0`, it returns the identity matrix or a batch of identity matrices.

    - If `n < 0`, it returns the inverse of each matrix (if invertible) raised to the power of `abs(n)`.

    Args:
        x (Tensor): A square matrix or a batch of square matrices to be raised
            to power `n`. Its shape should be `[*, M, M]`, where `*` is zero or
            more batch dimensions. Its data type should be float32 or float64.
        n (int): The exponent. It can be any positive, negative integer or zero.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        - Tensor, The n-th power of the matrix (or the batch of matrices) `x`. Its
          data type should be the same as that of `x`.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2, 3],
            ...                       [1, 4, 9],
            ...                       [1, 8, 27]], dtype='float64')
            >>> print(paddle.linalg.matrix_power(x, 2))
            Tensor(shape=[3, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[6.  , 34. , 102.],
             [14. , 90. , 282.],
             [36. , 250., 804.]])

            >>> print(paddle.linalg.matrix_power(x, 0))
            Tensor(shape=[3, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]])

            >>> print(paddle.linalg.matrix_power(x, -2))
            Tensor(shape=[3, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[ 12.91666667, -12.75000000,  2.83333333 ],
             [-7.66666667 ,  8.         , -1.83333333 ],
             [ 1.80555556 , -1.91666667 ,  0.44444444 ]])
    """
    if in_dynamic_mode():
        return _C_ops.matrix_power(x, n)
    else:
        check_variable_and_dtype(
            x, 'dtype', ['float32', 'float64'], 'matrix_power'
        )
        check_type(n, 'n', int, 'matrix_power')
        helper = LayerHelper('matrix_power', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='matrix_power',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'n': n},
        )
        return out


def qr(x, mode="reduced", name=None):
    r"""
    Computes the QR decomposition of one matrix or batches of matrice (backward is unsupported now).

    Args:
        x (Tensor): The input tensor. Its shape should be `[..., M, N]`,
            where ... is zero or more batch dimensions. M and N can be arbitrary
            positive number. The data type of x should be float32 or float64.
        mode (str, optional): A flag to control the behavior of qr.
            Suppose x's shape is `[..., M, N]` and denoting `K = min(M, N)`:
            If mode = "reduced", qr op will return reduced Q and R matrices,
            which means Q's shape is `[..., M, K]` and R's shape is `[..., K, N]`.
            If mode = "complete", qr op will return complete Q and R matrices,
            which means Q's shape is `[..., M, M]` and R's shape is `[..., M, N]`.
            If mode = "r", qr op will only return reduced R matrix, which means
            R's shape is `[..., K, N]`. Default: "reduced".
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        If mode = "reduced" or mode = "complete", qr will return a two tensor-tuple, which represents Q and R.
        If mode = "r", qr will return a tensor which represents R.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype('float64')
            >>> q, r = paddle.linalg.qr(x)
            >>> print (q)
            Tensor(shape=[3, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[-0.16903085,  0.89708523],
             [-0.50709255,  0.27602622],
             [-0.84515425, -0.34503278]])
            >>> print (r)
            Tensor(shape=[2, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[-5.91607978, -7.43735744],
             [ 0.        ,  0.82807867]])

            >>> # one can verify : X = Q * R ;
    """
    if in_dynamic_mode():
        q, r = _C_ops.qr(x, mode)
        if mode == "r":
            return r
        else:
            return q, r
    else:
        check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'qr')
        check_type(mode, 'mode', str, 'qr')
        helper = LayerHelper('qr', **locals())
        q = helper.create_variable_for_type_inference(dtype=x.dtype)
        r = helper.create_variable_for_type_inference(dtype=x.dtype)
        attrs = {}
        attrs['mode'] = mode
        helper.append_op(
            type='qr', inputs={'X': [x]}, outputs={'Q': q, 'R': r}, attrs=attrs
        )
        if mode == "r":
            return r
        else:
            return q, r


def lu(x, pivot=True, get_infos=False, name=None):
    r"""
    Computes the LU factorization of an N-D(N>=2) matrix x.

    Returns the LU factorization(inplace x) and Pivots. low triangular matrix L and
    upper triangular matrix U are combined to a single LU matrix.

    Pivoting is done if pivot is set to True.
    P mat can be get by pivots:

    .. code-block:: text

        ones = eye(rows) #eye matrix of rank rows
        for i in range(cols):
            swap(ones[i], ones[pivots[i]])
        return ones

    Args:

        X (Tensor): the tensor to factor of N-dimensions(N>=2).

        pivot (bool, optional): controls whether pivoting is done. Default: True.

        get_infos (bool, optional): if set to True, returns an info IntTensor. Default: False.

        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        factorization (Tensor), LU matrix, the factorization of input X.

        pivots (IntTensor), the pivots of size(∗(N-2), min(m,n)). `pivots` stores all the
        intermediate transpositions of rows. The final permutation `perm` could be
        reconstructed by this, details refer to upper example.

        infos (IntTensor, optional), if `get_infos` is `True`, this is a tensor of size (∗(N-2))
        where non-zero values indicate whether factorization for the matrix or each minibatch
        has succeeded or failed.


    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype('float64')
            >>> lu,p,info = paddle.linalg.lu(x, get_infos=True)

            >>> print(lu)
            Tensor(shape=[3, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[5.        , 6.        ],
             [0.20000000, 0.80000000],
             [0.60000000, 0.50000000]])
            >>> print(p)
            Tensor(shape=[2], dtype=int32, place=Place(cpu), stop_gradient=True,
            [3, 3])
            >>> print(info)
            Tensor(shape=[1], dtype=int32, place=Place(cpu), stop_gradient=True,
            [0])

            >>> P,L,U = paddle.linalg.lu_unpack(lu,p)

            >>> print(P)
            Tensor(shape=[3, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[0., 1., 0.],
             [0., 0., 1.],
             [1., 0., 0.]])
            >>> print(L)
            Tensor(shape=[3, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[1.        , 0.        ],
             [0.20000000, 1.        ],
             [0.60000000, 0.50000000]])
            >>> print(U)
            Tensor(shape=[2, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[5.        , 6.        ],
             [0.        , 0.80000000]])

            >>> # one can verify : X = P @ L @ U ;
    """

    if in_dynamic_mode():
        lu, p, info = _C_ops.lu(x, pivot)
    else:
        check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'lu')
        helper = LayerHelper('lu', **locals())
        lu = helper.create_variable_for_type_inference(dtype=x.dtype)
        p = helper.create_variable_for_type_inference(dtype='int')
        info = helper.create_variable_for_type_inference(dtype='int')
        attrs = {}
        attrs['pivot'] = pivot
        helper.append_op(
            type='lu',
            inputs={'X': x},
            outputs={'Out': lu, 'Pivots': p, 'Infos': info},
            attrs=attrs,
        )
    if get_infos:
        return lu, p, info
    else:
        return lu, p


def lu_unpack(x, y, unpack_ludata=True, unpack_pivots=True, name=None):
    r"""
    Unpack L U and P to single matrix tensor .
    unpack L and U matrix from LU, unpack permutation matrix P from Pivtos .

    P mat can be get by pivots:

    .. code-block:: text

        ones = eye(rows) #eye matrix of rank rows
        for i in range(cols):
            swap(ones[i], ones[pivots[i]])


    Args:
        x (Tensor): The LU tensor get from paddle.linalg.lu, which is combined by L and U.

        y (Tensor): Pivots get from paddle.linalg.lu.

        unpack_ludata (bool, optional): whether to unpack L and U from x. Default: True.

        unpack_pivots (bool, optional): whether to unpack permutation matrix P from Pivtos. Default: True.

        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        P (Tensor), Permutation matrix P of lu factorization.

        L (Tensor), The lower triangular matrix tensor of lu factorization.

        U (Tensor), The upper triangular matrix tensor of lu factorization.


    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype('float64')
            >>> lu,p,info = paddle.linalg.lu(x, get_infos=True)

            >>> print(lu)
            Tensor(shape=[3, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[5.        , 6.        ],
             [0.20000000, 0.80000000],
             [0.60000000, 0.50000000]])
            >>> print(p)
            Tensor(shape=[2], dtype=int32, place=Place(cpu), stop_gradient=True,
            [3, 3])
            >>> print(info)
            Tensor(shape=[1], dtype=int32, place=Place(cpu), stop_gradient=True,
            [0])

            >>> P,L,U = paddle.linalg.lu_unpack(lu,p)

            >>> print(P)
            Tensor(shape=[3, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[0., 1., 0.],
             [0., 0., 1.],
             [1., 0., 0.]])
            >>> print(L)
            Tensor(shape=[3, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[1.        , 0.        ],
             [0.20000000, 1.        ],
             [0.60000000, 0.50000000]])
            >>> print(U)
            Tensor(shape=[2, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[5.        , 6.        ],
             [0.        , 0.80000000]])

            >>> # one can verify : X = P @ L @ U ;
    """
    if x.ndim < 2:
        raise ValueError(
            f"The shape of x should be (*, M, N), but received ndim is [{x.ndim} < 2]"
        )
    if y.ndim < 1:
        raise ValueError(
            f"The shape of Pivots should be (*, K), but received ndim is [{y.ndim} < 1]"
        )
    if in_dynamic_mode():
        P, L, U = _C_ops.lu_unpack(x, y, unpack_ludata, unpack_pivots)
        return P, L, U
    else:
        check_variable_and_dtype(
            x, 'dtype', ['float32', 'float64'], 'lu_unpack'
        )
        helper = LayerHelper('lu_unpack', **locals())
        p = helper.create_variable_for_type_inference(dtype=x.dtype)
        l = helper.create_variable_for_type_inference(dtype=x.dtype)
        u = helper.create_variable_for_type_inference(dtype=x.dtype)

        attrs = {}
        attrs['unpack_ludata'] = unpack_ludata
        attrs['unpack_pivots'] = unpack_pivots
        helper.append_op(
            type='lu_unpack',
            inputs={'X': x, 'Pivots': y},
            outputs={'Pmat': p, 'L': l, 'U': u},
            attrs=attrs,
        )
        return p, l, u


def eig(x, name=None):
    """
    Performs the eigenvalue decomposition of a square matrix or a batch of square matrices.

    Note:
        - If the matrix is a Hermitian or a real symmetric matrix, please use :ref:`api_paddle_linalg_eigh` instead, which is much faster.
        - If only eigenvalues is needed, please use :ref:`api_paddle_linalg_eigvals` instead.
        - If the matrix is of any shape, please use :ref:`api_paddle_linalg_svd`.
        - This API is only supported on CPU device.
        - The output datatype is always complex for both real and complex input.

    Args:
        x (Tensor): A tensor with shape math:`[*, N, N]`, The data type of the x should be one of ``float32``,
            ``float64``, ``compplex64`` or ``complex128``.
        name (str, optional): The default value is `None`. Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Eigenvalues(Tensors): A tensor with shape math:`[*, N]` refers to the eigen values.
        Eigenvectors(Tensors): A tensor with shape math:`[*, N, N]` refers to the eigen vectors.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.6707249, 7.2249975, 6.5045543],
            ...                       [9.956216,  8.749598,  6.066444 ],
            ...                       [4.4251957, 1.7983172, 0.370647 ]])
            >>> w, v = paddle.linalg.eig(x)
            >>> print(v)
            Tensor(shape=[3, 3], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[ (0.5061365365982056+0j) ,  (0.7971761226654053+0j) ,
               (0.1851806491613388+0j) ],
             [ (0.8308236598968506+0j) , (-0.3463813066482544+0j) ,
               (-0.6837005615234375+0j) ],
             [ (0.23142573237419128+0j), (-0.49449989199638367+0j),
               (0.7058765292167664+0j) ]])

            >>> print(w)
            Tensor(shape=[3], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [ (16.50470733642578+0j)  , (-5.503481388092041+0j)  ,
              (-0.21026138961315155+0j)])
    """

    if in_dynamic_mode():
        return _C_ops.eig(x)
    else:
        check_variable_and_dtype(
            x, 'X', ['float32', 'float64', 'complex64', 'complex128'], 'eig'
        )
        helper = LayerHelper('eig', **locals())

        w = helper.create_variable_for_type_inference(x.dtype)
        v = helper.create_variable_for_type_inference(x.dtype)

        inputs = {'X': x}
        outputs = {'Eigenvalues': w, 'Eigenvectors': v}
        helper.append_op(type='eig', inputs=inputs, outputs=outputs)

        return w, v


def eigvals(x, name=None):
    """
    Compute the eigenvalues of one or more general matrices.

    Warning:
        The gradient kernel of this operator does not yet developed.
        If you need back propagation through this operator, please replace it with paddle.linalg.eig.

    Args:
        x (Tensor): A square matrix or a batch of square matrices whose eigenvalues will be computed.
            Its shape should be `[*, M, M]`, where `*` is zero or more batch dimensions.
            Its data type should be float32, float64, complex64, or complex128.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, A tensor containing the unsorted eigenvalues which has the same batch
        dimensions with `x`. The eigenvalues are complex-valued even when `x` is real.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> x = paddle.rand(shape=[3, 3], dtype='float64')
            >>> print(x)
            Tensor(shape=[3, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[0.86583615, 0.52014721, 0.25960938],
             [0.90525323, 0.42400090, 0.40641288],
             [0.97020893, 0.74437359, 0.51785128]])

            >>> print(paddle.linalg.eigvals(x))
            Tensor(shape=[3], dtype=complex128, place=Place(cpu), stop_gradient=True,
            [ (1.788956694280852+0j)  ,  (0.16364484879581526+0j),
              (-0.14491322408727625+0j)])
    """

    x_shape = list(x.shape)
    if len(x_shape) < 2:
        raise ValueError(
            "The dimension of Input(x) should be at least 2, but received x's dimention = {}, x's shape = {}".format(
                len(x_shape), x_shape
            )
        )

    if x_shape[-1] != x_shape[-2]:
        raise ValueError(
            "The last two dimensions of Input(x) should be equal, but received x's shape = {}".format(
                x_shape
            )
        )

    if in_dynamic_mode():
        return _C_ops.eigvals(x)
    else:
        check_variable_and_dtype(
            x,
            'dtype',
            ['float32', 'float64', 'complex64', 'complex128'],
            'eigvals',
        )
        helper = LayerHelper('eigvals', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='eigvals', inputs={'X': x}, outputs={'Out': out})
        return out


def multi_dot(x, name=None):
    """
    Multi_dot is an operator that calculates multiple matrix multiplications.

    Supports inputs of float16(only GPU support), float32 and float64 dtypes. This function does not
    support batched inputs.

    The input tensor in [x] must be 2-D except for the first and last can be 1-D.
    If the first tensor is a 1-D vector of shape(n, ) it is treated as row vector
    of shape(1, n), similarly if the last tensor is a 1D vector of shape(n, ), it
    is treated as a column vector of shape(n, 1).

    If the first and last tensor are 2-D matrix, then the output is also 2-D matrix,
    otherwise the output is a 1-D vector.

    Multi_dot will select the lowest cost multiplication order for calculation. The
    cost of multiplying two matrices with shapes (a, b) and (b, c) is a * b * c.
    Given matrices A, B, C with shapes (20, 5), (5, 100), (100, 10) respectively,
    we can calculate the cost of different multiplication orders as follows:
    - Cost((AB)C) = 20x5x100 + 20x100x10 = 30000
    - Cost(A(BC)) = 5x100x10 + 20x5x10 = 6000

    In this case, multiplying B and C first, then multiply A, which is 5 times faster
    than sequential calculation.

    Args:
        x ([Tensor]): The input tensors which is a list Tensor.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output Tensor.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> # A * B
            >>> A = paddle.rand([3, 4])
            >>> B = paddle.rand([4, 5])
            >>> out = paddle.linalg.multi_dot([A, B])
            >>> print(out.shape)
            [3, 5]

            >>> # A * B * C
            >>> A = paddle.rand([10, 5])
            >>> B = paddle.rand([5, 8])
            >>> C = paddle.rand([8, 7])
            >>> out = paddle.linalg.multi_dot([A, B, C])
            >>> print(out.shape)
            [10, 7]

    """
    if in_dynamic_mode():
        return _C_ops.multi_dot(x)
    else:
        check_type(x, 'x', (list, tuple), 'multi_dot')
        for id, item in enumerate(x):
            check_variable_and_dtype(
                item,
                'x[' + str(id) + ']',
                ['float16', 'float32', 'float64', 'uint16'],
                'multi_dot',
            )
            if item.dtype != x[0].dtype:
                raise TypeError(
                    "All the Tensors in the input must have the same data type."
                )

        helper = LayerHelper('multi_dot', **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='multi_dot', inputs={"X": x}, outputs={"Out": out}
        )
        return out


def eigh(x, UPLO='L', name=None):
    """
    Compute the eigenvalues and eigenvectors of a
    complex Hermitian (conjugate symmetric) or a real symmetric matrix.

    Args:
        x (Tensor): A tensor with shape :math:`[*, N, N]` , The data type of the input Tensor x
            should be one of float32, float64, complex64, complex128.
        UPLO (str, optional): (string, default 'L'), 'L' represents the lower triangular matrix,
            "'U' represents the upper triangular matrix.". Default: 'L'.
        name (str, optional): The default value is None. Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        - out_value(Tensor):  A Tensor with shape [*, N] and data type of float32 and float64.
            The eigenvalues of eigh op.
        - out_vector(Tensor): A Tensor with shape [*, N, N] and data type of float32,float64,
            complex64 and complex128. The eigenvectors of eigh op.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, -2j], [2j, 5]])
            >>> out_value, out_vector = paddle.linalg.eigh(x, UPLO='L')
            >>> print(out_value)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.17157286, 5.82842731])
            >>> print(out_vector)
            Tensor(shape=[2, 2], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[(-0.9238795042037964+0j), (-0.3826833963394165+0j)],
             [ 0.3826833963394165j    , -0.9238795042037964j    ]])

    """
    if in_dynamic_mode():
        return _C_ops.eigh(x, UPLO)
    else:

        def __check_input(x, UPLO):
            x_shape = list(x.shape)
            if len(x.shape) < 2:
                raise ValueError(
                    "Input(input) only support >=2 tensor, but received "
                    "length of Input(input) is %s." % len(x.shape)
                )
            if x_shape[-1] != x_shape[-2]:
                raise ValueError(
                    "The input matrix must be batches of square matrices. But received x's dimention: {}".format(
                        x_shape
                    )
                )
            if UPLO != 'L' and UPLO != 'U':
                raise ValueError(
                    f"UPLO must be L or U. But received UPLO is: {UPLO}"
                )

        __check_input(x, UPLO)

        helper = LayerHelper('eigh', **locals())
        check_variable_and_dtype(
            x,
            'dtype',
            ['float32', 'float64', 'complex64', 'complex128'],
            'eigh',
        )

        out_value = helper.create_variable_for_type_inference(dtype=x.dtype)
        out_vector = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='eigh',
            inputs={'X': x},
            outputs={'Eigenvalues': out_value, 'Eigenvectors': out_vector},
            attrs={'UPLO': UPLO},
        )
        return out_value, out_vector


def pinv(x, rcond=1e-15, hermitian=False, name=None):
    r"""
    Calculate pseudo inverse via SVD(singular value decomposition)
    of one matrix or batches of regular matrix.

    .. math::

        if hermitian == False:
            x = u * s * vt  (SVD)
            out = v * 1/s * ut
        else:
            x = u * s * ut  (eigh)
            out = u * 1/s * u.conj().transpose(-2,-1)

    If x is hermitian or symmetric matrix, svd will be replaced with eigh.

    Args:
        x (Tensor): The input tensor. Its shape should be (*, m, n)
            where * is zero or more batch dimensions. m and n can be
            arbitraty positive number. The data type of x should be
            float32 or float64 or complex64 or complex128. When data
            type is complex64 or cpmplex128, hermitian should be set
            True.
        rcond (Tensor, optional): the tolerance value to determine
            when is a singular value zero. Default:1e-15.
        hermitian (bool, optional): indicates whether x is Hermitian
            if complex or symmetric if real. Default: False.
        name (str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The tensor with same data type with x. it represents
        pseudo inverse of x. Its shape should be (*, n, m).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.arange(15).reshape((3, 5)).astype('float64')
            >>> input = paddle.to_tensor(x)
            >>> out = paddle.linalg.pinv(input)
            >>> print(input)
            Tensor(shape=[3, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[0. , 1. , 2. , 3. , 4. ],
             [5. , 6. , 7. , 8. , 9. ],
             [10., 11., 12., 13., 14.]])

            >>> print(out)
            Tensor(shape=[5, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[-0.22666667, -0.06666667,  0.09333333],
             [-0.12333333, -0.03333333,  0.05666667],
             [-0.02000000, -0.00000000,  0.02000000],
             [ 0.08333333,  0.03333333, -0.01666667],
             [ 0.18666667,  0.06666667, -0.05333333]])

            # one can verify : x * out * x = x ;
            # or              out * x * out = x ;
    """
    if in_dynamic_mode():
        if not hermitian:
            # combine svd and matmul op
            u, s, vt = _C_ops.svd(x, False)
            max_singular_val = _C_ops.max(s, [-1], True)
            rcond = paddle.to_tensor(rcond, dtype=x.dtype)
            cutoff = rcond * max_singular_val
            y = float('inf')
            y = paddle.to_tensor(y, dtype=x.dtype)

            singular = paddle.where(s > cutoff, 1 / s, 1 / y)
            st = _C_ops.unsqueeze(singular, [-2])

            dims = list(range(len(vt.shape)))
            perm = dims[:-2] + [dims[-1]] + [dims[-2]]
            v = _C_ops.transpose(vt, perm)

            out_1 = v * st
            out_2 = _C_ops.matmul(out_1, u, False, True)
            return out_2
        else:
            # combine eigh and matmul op
            s, u = _C_ops.eigh(x, 'UPLO')
            s_abs = paddle.abs(s)
            max_singular_val = _C_ops.max(s_abs, [-1], True)
            rcond = paddle.to_tensor(rcond, dtype=s.dtype)
            cutoff = rcond * max_singular_val
            y = float('inf')
            y = paddle.to_tensor(y, dtype=s.dtype)

            singular = paddle.where(s_abs > cutoff, 1 / s, 1 / y)
            st = _C_ops.unsqueeze(singular, [-2])

            out_1 = u * st
            u_conj = _C_ops.conj(u)
            out_2 = _C_ops.matmul(out_1, u_conj, False, True)
            return out_2
    else:
        if not hermitian:
            helper = LayerHelper('pinv', **locals())
            dtype = x.dtype
            check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'pinv')

            u = helper.create_variable_for_type_inference(dtype)
            s = helper.create_variable_for_type_inference(dtype)
            vt = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='svd',
                inputs={'X': [x]},
                outputs={'U': u, 'VH': vt, 'S': s},
                attrs={'full_matrices': False},
            )

            max_singular_val = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='reduce_max',
                inputs={'X': s},
                outputs={'Out': max_singular_val},
                attrs={'dim': [-1], 'keep_dim': True, 'reduce_all': False},
            )

            rcond = full(shape=[1], fill_value=rcond, dtype=dtype)
            cutoff = rcond * max_singular_val
            y = float('inf')
            y = full(shape=[1], fill_value=y, dtype=dtype)

            singular = paddle.where(s > cutoff, 1 / s, 1 / y)

            st = helper.create_variable_for_type_inference(dtype=dtype)
            st_shape = helper.create_variable_for_type_inference(dtype=dtype)
            helper.append_op(
                type='unsqueeze2',
                inputs={'X': singular},
                attrs={'axes': [-2]},
                outputs={'Out': st, 'XShape': st_shape},
            )

            dims = list(range(len(vt.shape)))
            perm = dims[:-2] + [dims[-1]] + [dims[-2]]
            v = helper.create_variable_for_type_inference(dtype)
            v_shape = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='transpose2',
                inputs={'X': [vt]},
                outputs={'Out': [v], 'XShape': [v_shape]},
                attrs={'axis': perm},
            )

            out_1 = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='elementwise_mul',
                inputs={'X': v, 'Y': st},
                outputs={'Out': out_1},
                attrs={'axis': -1, 'use_mkldnn': False},
            )
            out_1 = helper.append_activation(out_1)

            out_2 = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='matmul_v2',
                inputs={'X': out_1, 'Y': u},
                outputs={'Out': out_2},
                attrs={'trans_x': False, 'trans_y': True},
            )
            return out_2
        else:
            helper = LayerHelper('pinv', **locals())
            dtype = x.dtype
            check_variable_and_dtype(
                x,
                'dtype',
                ['float32', 'float64', 'complex64', 'complex128'],
                'pinv',
            )

            if dtype == paddle.complex128:
                s_type = 'float64'
            elif dtype == paddle.complex64:
                s_type = 'float32'
            else:
                s_type = dtype

            u = helper.create_variable_for_type_inference(dtype)
            s = helper.create_variable_for_type_inference(s_type)
            helper.append_op(
                type='eigh',
                inputs={'X': x},
                outputs={'Eigenvalues': s, 'Eigenvectors': u},
                attrs={'UPLO': 'L'},
            )
            s_abs = helper.create_variable_for_type_inference(s_type)
            helper.append_op(
                type='abs', inputs={'X': s}, outputs={'Out': s_abs}
            )
            max_singular_val = helper.create_variable_for_type_inference(s_type)
            helper.append_op(
                type='reduce_max',
                inputs={'X': s_abs},
                outputs={'Out': max_singular_val},
                attrs={'dim': [-1], 'keep_dim': True, 'reduce_all': False},
            )

            rcond = full(shape=[1], fill_value=rcond, dtype=s_type)
            cutoff = rcond * max_singular_val
            y = float('inf')
            y = full(shape=[1], fill_value=y, dtype=s_type)

            singular = paddle.where(s_abs > cutoff, 1 / s, 1 / y)

            st = helper.create_variable_for_type_inference(dtype=s_type)
            st_shape = helper.create_variable_for_type_inference(dtype=s_type)
            helper.append_op(
                type='unsqueeze2',
                inputs={'X': singular},
                attrs={'axes': [-2]},
                outputs={'Out': st, 'XShape': st_shape},
            )

            out_1 = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='elementwise_mul',
                inputs={'X': u, 'Y': st},
                outputs={'Out': out_1},
                attrs={'axis': -1, 'use_mkldnn': False},
            )
            out_1 = helper.append_activation(out_1)

            u_conj = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='conj', inputs={'X': u}, outputs={'Out': [u_conj]}
            )

            out_2 = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='matmul_v2',
                inputs={'X': out_1, 'Y': u_conj},
                outputs={'Out': out_2},
                attrs={'trans_x': False, 'trans_y': True},
            )
            return out_2


def solve(x, y, name=None):
    r"""

    Computes the solution of a square system of linear equations with a unique solution for input 'X' and 'Y'.
    Let :math:`X` be a sqaure matrix or a batch of square matrices, :math:`Y` be
    a vector/matrix or a batch of vectors/matrices, the equation should be:

    .. math::
        Out = X^-1 * Y

    Specifically, this system of linear equations has one solution if and only if input 'X' is invertible.

    Args:
        x (Tensor): A square matrix or a batch of square matrices. Its shape should be ``[*, M, M]``, where ``*`` is zero or
            more batch dimensions. Its data type should be float32 or float64.
        y (Tensor): A vector/matrix or a batch of vectors/matrices. Its shape should be ``[*, M, K]``, where ``*`` is zero or
            more batch dimensions. Its data type should be float32 or float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The solution of a square system of linear equations with a unique solution for input 'x' and 'y'.
        Its data type should be the same as that of `x`.

    Examples:

        .. code-block:: python

            >>> # a square system of linear equations:
            >>> # 2*X0 + X1 = 9
            >>> # X0 + 2*X1 = 8

            >>> import paddle

            >>> x = paddle.to_tensor([[3, 1],[1, 2]], dtype="float64")
            >>> y = paddle.to_tensor([9, 8], dtype="float64")
            >>> out = paddle.linalg.solve(x, y)

            >>> print(out)
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [2., 3.])
    """
    if in_dynamic_mode():
        return _C_ops.solve(x, y)
    else:
        inputs = {"X": [x], "Y": [y]}
        helper = LayerHelper("solve", **locals())
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'solve')
        check_variable_and_dtype(y, 'y', ['float32', 'float64'], 'solve')
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type="solve", inputs={"X": x, "Y": y}, outputs={"Out": out}
        )
        return out


def triangular_solve(
    x, y, upper=True, transpose=False, unitriangular=False, name=None
):
    r"""
    Computes the solution of a system of equations with a triangular coefficient. `x` is coefficient matrix
    `y` is multiple right-hand sides of equations.

    Input `x` and `y` is 2D matrices or batches of 2D matrices. If the inputs are batches, the outputs is also
    batches.

    Equations can be described as:

    .. math::
        x * Out = y

    Solution of Equations is:

    .. math::
        Out = x ^ {-1} * y

    Args:
        x (Tensor): The input triangular coefficient matrix. Its shape should be `[*, M, M]`, where `*` is zero or
            more batch dimensions. Its data type should be float32 or float64.
        y (Tensor): Multiple right-hand sides of system of equations. Its shape should be `[*, M, K]`, where `*` is
            zero or more batch dimensions. Its data type should be float32 or float64.
        upper (bool, optional): Whether to solve the upper-triangular system of equations (default) or the lower-triangular
            system of equations. Default: True.
        transpose (bool, optional): whether `x` should be transposed before calculation. Default: False.
        unitriangular (bool, optional): whether `x` is unit triangular. If True, the diagonal elements of `x` are assumed
            to be 1 and not referenced from `x` . Default: False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The solution of the system of equations. Its data type should be the same as that of `x`.

    Examples:
        .. code-block:: python

            >>> # a square system of linear equations:
            >>> # x1 +   x2  +   x3 = 0
            >>> #      2*x2  +   x3 = -9
            >>> #               -x3 = 5

            >>> import paddle
            >>> x = paddle.to_tensor([[1, 1, 1],
            ...                       [0, 2, 1],
            ...                       [0, 0,-1]], dtype="float64")
            >>> y = paddle.to_tensor([[0], [-9], [5]], dtype="float64")
            >>> out = paddle.linalg.triangular_solve(x, y, upper=True)

            >>> print(out)
            Tensor(shape=[3, 1], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[ 7.],
             [-2.],
             [-5.]])
    """
    if in_dynamic_mode():
        return _C_ops.triangular_solve(x, y, upper, transpose, unitriangular)
    else:
        inputs = {"X": [x], "Y": [y]}
        helper = LayerHelper("triangular_solve", **locals())
        check_variable_and_dtype(
            x, 'x', ['float32', 'float64'], 'triangular_solve'
        )
        check_variable_and_dtype(
            y, 'y', ['float32', 'float64'], 'triangular_solve'
        )
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='triangular_solve',
            inputs={'X': x, 'Y': y},
            outputs={'Out': out},
            attrs={
                'upper': upper,
                'transpose': transpose,
                'unitriangular': unitriangular,
            },
        )
        return out


def cholesky_solve(x, y, upper=False, name=None):
    r"""
    Solves a linear system of equations A @ X = B, given A's Cholesky factor matrix u and  matrix B.

    Input `x` and `y` is 2D matrices or batches of 2D matrices. If the inputs are batches, the outputs
    is also batches.

    Args:
        x (Tensor): Multiple right-hand sides of system of equations. Its shape should be `[*, M, K]`, where `*` is
            zero or more batch dimensions. Its data type should be float32 or float64.
        y (Tensor): The input matrix which is upper or lower triangular Cholesky factor of square matrix A. Its shape should be `[*, M, M]`, where `*` is zero or
            more batch dimensions. Its data type should be float32 or float64.
        upper (bool, optional): whether to consider the Cholesky factor as a lower or upper triangular matrix. Default: False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The solution of the system of equations. Its data type is the same as that of `x`.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> u = paddle.to_tensor([[1, 1, 1],
            ...                       [0, 2, 1],
            ...                       [0, 0,-1]], dtype="float64")
            >>> b = paddle.to_tensor([[0], [-9], [5]], dtype="float64")
            >>> out = paddle.linalg.cholesky_solve(b, u, upper=True)

            >>> print(out)
            Tensor(shape=[3, 1], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[-2.50000000],
             [-7.        ],
             [ 9.50000000]])
    """
    if in_dynamic_mode():
        return _C_ops.cholesky_solve(x, y, upper)
    else:
        helper = LayerHelper("cholesky_solve", **locals())
        check_variable_and_dtype(
            x, 'x', ['float32', 'float64'], 'cholesky_solve'
        )
        check_variable_and_dtype(
            y, 'y', ['float32', 'float64'], 'cholesky_solve'
        )
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='cholesky_solve',
            inputs={'X': x, 'Y': y},
            outputs={'Out': out},
            attrs={'upper': upper},
        )
        return out


def eigvalsh(x, UPLO='L', name=None):
    """
    Computes the eigenvalues of a
    complex Hermitian (conjugate symmetric) or a real symmetric matrix.

    Args:
        x (Tensor): A tensor with shape :math:`[*, M, M]` , where * is zero or greater batch dimension. The data type of the input Tensor x
            should be one of float32, float64, complex64, complex128.
        UPLO(str, optional): Lower triangular part of a (‘L’, default) or the upper triangular part (‘U’).
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The tensor eigenvalues in ascending order.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, -2j], [2j, 5]])
            >>> out_value = paddle.eigvalsh(x, UPLO='L')
            >>> print(out_value)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.17157286, 5.82842731])
    """
    if in_dynamic_mode():
        values, _ = _C_ops.eigvalsh(x, UPLO, x.stop_gradient)
        return values
    else:

        def __check_input(x, UPLO):
            x_shape = list(x.shape)
            if len(x.shape) < 2:
                raise ValueError(
                    "Input(input) only support >=2 tensor, but received "
                    "length of Input(input) is %s." % len(x.shape)
                )
            if x_shape[-1] != x_shape[-2]:
                raise ValueError(
                    "The input matrix must be batches of square matrices. But received x's dimention: {}".format(
                        x_shape
                    )
                )
            if UPLO != 'L' and UPLO != 'U':
                raise ValueError(
                    f"UPLO must be L or U. But received UPLO is: {UPLO}"
                )

        __check_input(x, UPLO)

        helper = LayerHelper('eigvalsh', **locals())
        check_variable_and_dtype(
            x,
            'dtype',
            ['float32', 'float64', 'complex64', 'complex128'],
            'eigvalsh',
        )

        out_value = helper.create_variable_for_type_inference(dtype=x.dtype)
        out_vector = helper.create_variable_for_type_inference(dtype=x.dtype)

        is_test = x.stop_gradient
        helper.append_op(
            type='eigvalsh',
            inputs={'X': x},
            outputs={'Eigenvalues': out_value, 'Eigenvectors': out_vector},
            attrs={'UPLO': UPLO, 'is_test': is_test},
        )
        return out_value


def lstsq(x, y, rcond=None, driver=None, name=None):
    """
    Computes a solution to
    the least squares problem of a system of linear equations.

    Args:
        x (Tensor): A tensor with shape ``(*, M, N)`` , the data type of the input Tensor ``x``
            should be one of float32, float64.
        y (Tensor): A tensor with shape ``(*, M, K)`` , the data type of the input Tensor ``y``
            should be one of float32, float64.
        rcond(float, optional): The default value is None. A float pointing number used to determine
            the effective rank of ``x``. If ``rcond`` is None, it will be set to max(M, N) times the
            machine precision of x_dtype.
        driver(str, optional): The default value is None. The name of LAPACK method to be used. For
            CPU inputs the valid values are ‘gels’, ‘gelsy’, ‘gelsd, ‘gelss’. For CUDA input, the only
            valid driver is ‘gels’. If ``driver`` is None, ‘gelsy’ is used for CPU inputs and ‘gels’
            for CUDA inputs.
        name(str, optional): The default value is None. Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tuple: A tuple of 4 Tensors which is (``solution``, ``residuals``, ``rank``, ``singular_values``).
        ``solution`` is a tensor with shape ``(*, N, K)``, meaning the least squares solution. ``residuals``
        is a tensor with shape ``(*, K)``, meaning the squared residuals of the solutions, which is computed
        when M > N and every matrix in ``x`` is full-rank, otherwise return an empty tensor. ``rank`` is a tensor
        with shape ``(*)``, meaning the ranks of the matrices in ``x``, which is computed when ``driver`` in
        (‘gelsy’, ‘gelsd’, ‘gelss’), otherwise return an empty tensor. ``singular_values`` is a tensor with
        shape ``(*, min(M, N))``, meaning singular values of the matrices in ``x``, which is computed when
        ``driver`` in (‘gelsd’, ‘gelss’), otherwise return an empty tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 3], [3, 2], [5, 6.]])
            >>> y = paddle.to_tensor([[3, 4, 6], [5, 3, 4], [1, 2, 1.]])
            >>> results = paddle.linalg.lstsq(x, y, driver="gelsd")
            >>> print(results[0])
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.78350395, -0.22165027, -0.62371236],
             [-0.11340097,  0.78866047,  1.14948535]])
            >>> print(results[1])
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [19.81443405, 10.43814468, 30.56185532])
            >>> print(results[2])
            Tensor(shape=[], dtype=int32, place=Place(cpu), stop_gradient=True,
            2)
            >>> print(results[3])
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [9.03455734, 1.54167950])

            >>> x = paddle.to_tensor([[10, 2, 3], [3, 10, 5], [5, 6, 12.]])
            >>> y = paddle.to_tensor([[4, 2, 9], [2, 0, 3], [2, 5, 3.]])
            >>> results = paddle.linalg.lstsq(x, y, driver="gels")
            >>> print(results[0])
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.39386186,  0.10230169,  0.93606132],
             [ 0.10741688, -0.29028130,  0.11892584],
             [-0.05115093,  0.51918161, -0.19948851]])
            >>> print(results[1])
            Tensor(shape=[0], dtype=float32, place=Place(cpu), stop_gradient=True,
            [])
    """
    device = paddle.get_device()
    if device == "cpu":
        if driver not in (None, "gels", "gelss", "gelsd", "gelsy"):
            raise ValueError(
                "Only support valid driver is 'gels', 'gelss', 'gelsd', 'gelsy' or None for CPU inputs. But got {}".format(
                    driver
                )
            )
        driver = "gelsy" if driver is None else driver
    elif "gpu" in device:
        if driver not in (None, "gels"):
            raise ValueError(
                "Only support valid driver is 'gels' or None for CUDA inputs. But got {}".format(
                    driver
                )
            )
        driver = "gels" if driver is None else driver
    else:
        raise RuntimeError("Only support lstsq api for CPU or CUDA device.")

    if not (x.dtype == y.dtype and x.dtype in (paddle.float32, paddle.float64)):
        raise ValueError(
            "Only support x and y have the same dtype such as 'float32' and 'float64'."
        )

    if x.ndim < 2:
        raise ValueError(
            f"The shape of x should be (*, M, N), but received ndim is [{x.ndim} < 2]"
        )

    if y.ndim < 2:
        raise ValueError(
            f"The shape of y should be (*, M, K), but received ndim is [{y.ndim} < 2]"
        )

    if x.shape[-2] != y.shape[-2]:
        raise ValueError(
            f"x with shape (*, M = {x.shape[-2]}, N) and y with shape (*, M = {y.shape[-2]}, K) should have same M."
        )

    if rcond is None:
        if x.dtype == paddle.float32:
            rcond = 1e-7 * max(x.shape[-2], x.shape[-1])
        elif x.dtype == paddle.float64:
            rcond = 1e-15 * max(x.shape[-2], x.shape[-1])

    if in_dynamic_mode():
        solution, residuals, rank, singular_values = _C_ops.lstsq(
            x, y, rcond, driver
        )
        if driver == "gels":
            rank = paddle.empty(shape=[0], dtype=paddle.int32)
            singular_values = paddle.empty(shape=[0], dtype=x.dtype)
        elif driver == "gelsy":
            singular_values = paddle.empty(shape=[0], dtype=x.dtype)

        return solution, residuals, rank, singular_values
    else:
        helper = LayerHelper('lstsq', **locals())
        check_variable_and_dtype(
            x,
            'dtype',
            ['float32', 'float64', 'complex64', 'complex128'],
            'lstsq',
        )
        check_variable_and_dtype(
            y,
            'dtype',
            ['float32', 'float64', 'complex64', 'complex128'],
            'lstsq',
        )

        solution = helper.create_variable_for_type_inference(dtype=x.dtype)
        residuals = helper.create_variable_for_type_inference(dtype=x.dtype)
        rank = helper.create_variable_for_type_inference(dtype=paddle.int32)
        singular_values = helper.create_variable_for_type_inference(
            dtype=x.dtype
        )

        helper.append_op(
            type='lstsq',
            inputs={'X': x, 'Y': y},
            outputs={
                'Solution': solution,
                'Residuals': residuals,
                'Rank': rank,
                'SingularValues': singular_values,
            },
            attrs={'rcond': rcond, 'driver': driver},
        )

        if driver == "gels":
            rank = paddle.static.data(name='rank', shape=[0])
            singular_values = paddle.static.data(
                name='singular_values', shape=[0]
            )
        elif driver == "gelsy":
            singular_values = paddle.static.data(
                name='singular_values', shape=[0]
            )

        return solution, residuals, rank, singular_values


def corrcoef(x, rowvar=True, name=None):
    """

    A correlation coefficient matrix indicate the correlation of each pair variables in the input matrix.
    For example, for an N-dimensional samples X=[x1,x2,…xN]T, then the correlation coefficient matrix
    element Rij is the correlation of xi and xj. The element Rii is the covariance of xi itself.

    The relationship between the correlation coefficient matrix `R` and the
    covariance matrix `C`, is

    .. math:: R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} * C_{jj} } }

    The values of `R` are between -1 and 1.

    Args:

        x (Tensor): A N-D(N<=2) Tensor containing multiple variables and observations. By default, each row of x represents a variable. Also see rowvar below.
        rowvar (bool, optional): If rowvar is True (default), then each row represents a variable, with observations in the columns. Default: True.
        name (str, optional): Name of the output. It's used to print debug info for developers. Details: :ref:`api_guide_Name`. Default: None.

    Returns:

        The correlation coefficient matrix of the variables.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> xt = paddle.rand((3,4))
            >>> print(paddle.linalg.corrcoef(xt))
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.99999988, -0.47689581, -0.89559376],
             [-0.47689593,  1.        ,  0.16345492],
             [-0.89559382,  0.16345496,  1.        ]])

    """
    if len(x.shape) > 2 or len(x.shape) < 1:
        raise ValueError(
            "Input(x) only support N-D (1<=N<=2) tensor in corrcoef, but received "
            "length of Input(input) is %s." % len(x.shape)
        )
    check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'corrcoef')

    c = cov(x, rowvar)
    if c.ndim == 0:
        # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return c / c

    d = paddle.diag(c)

    if paddle.is_complex(d):
        d = d.real()
    stddev = paddle.sqrt(d)
    c /= stddev[:, None]
    c /= stddev[None, :]

    # Clip to [-1, 1].  This does not guarantee
    if paddle.is_complex(c):
        return paddle.complex(
            paddle.clip(c.real(), -1, 1), paddle.clip(c.imag(), -1, 1)
        )
    else:
        c = paddle.clip(c, -1, 1)

    return c


def cdist(
    x, y, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary", name=None
):
    r"""

    Compute the p-norm distance between each pair of the two collections of inputs.

    This function is equivalent to `scipy.spatial.distance.cdist(input,'minkowski', p=p)`
    if :math:`p \in (0, \infty)`. When :math:`p = 0` it is equivalent to `scipy.spatial.distance.cdist(input, 'hamming') * M`.
    When :math:`p = \infty`, the closest scipy function is `scipy.spatial.distance.cdist(xn, lambda x, y: np.abs(x - y).max())`.

    Args:
        x (Tensor): A tensor with shape :math:`B \times P \times M`.
        y (Tensor): A tensor with shape :math:`B \times R \times M`.
        p (float, optional): The value for the p-norm distance to calculate between each vector pair. Default: :math:`2.0`.
        compute_mode (str, optional): The mode for compute distance.

            - ``use_mm_for_euclid_dist_if_necessary`` , for p = 2.0 and (P > 25 or R > 25), it will use matrix multiplication to calculate euclid distance if possible.
            - ``use_mm_for_euclid_dist`` , for p = 2.0, it will use matrix multiplication to calculate euclid distance.
            - ``donot_use_mm_for_euclid_dist`` , it will not use matrix multiplication to calculate euclid distance.

            Default: ``use_mm_for_euclid_dist_if_necessary``.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, the dtype is same as input tensor.

        If x has shape :math:`B \times P \times M` and y has shape :math:`B \times R \times M` then
        the output will have shape :math:`B \times P \times R`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]], dtype=paddle.float32)
            >>> y = paddle.to_tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]], dtype=paddle.float32)
            >>> distance = paddle.cdist(x, y)
            >>> print(distance)
            Tensor(shape=[3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[3.11927032, 2.09589314],
             [2.71384072, 3.83217239],
             [2.28300953, 0.37910119]])
    """

    check_variable_and_dtype(x, 'x', ('float32', 'float64'), 'cdist')
    check_variable_and_dtype(y, 'y', ('float32', 'float64'), 'cdist')
    check_type(p, 'p', (float, int), 'cdist')

    if compute_mode not in [
        'use_mm_for_euclid_dist_if_necessary',
        'use_mm_for_euclid_dist',
        'donot_use_mm_for_euclid_dist',
    ]:
        raise ValueError(
            "The compute_mode should be 'use_mm_for_euclid_dist_if_necessary', "
            "'use_mm_for_euclid_dist' or 'donot_use_mm_for_euclid_dist', "
            "but received compute_mode is %s." % compute_mode
        )

    mode = 0
    if compute_mode == 'use_mm_for_euclid_dist_if_necessary':
        mode = 0
    elif compute_mode == 'use_mm_for_euclid_dist':
        mode = 1
    elif compute_mode == 'donot_use_mm_for_euclid_dist':
        mode = 2

    x_shape = list(x.shape)
    assert len(x_shape) >= 2, (
        "The x must be at least 2-dimensional, "
        "But received Input x's dimensional is %s.\n" % len(x_shape)
    )
    y_shape = list(y.shape)
    assert len(y_shape) >= 2, (
        "The y must be at least 2-dimensional, "
        "But received Input y's dimensional is %s.\n" % len(y_shape)
    )
    assert x_shape[-1] == y_shape[-1], (
        "The x and y must have same last dimension, "
        f"But received Input x's last dimension is {x_shape[-1]}, "
        f"Input y's last dimension is {y_shape[-1]}.\n"
    )
    assert p >= 0, (
        "The p must be greater than or equal to 0, "
        "But received p is %s.\n" % p
    )

    r1 = x.shape[-2]
    r2 = y.shape[-2]
    c1 = x.shape[-1]

    p = float(p)

    if r1 == 0 or r2 == 0:
        return paddle.empty((r1, r2), dtype=x.dtype)

    if c1 == 0:
        return paddle.zeros((r1, r2), dtype=x.dtype)

    if p == 2.0 and (mode == 1 or (mode == 0 and (r1 > 25 or r2 > 25))):
        x_norm = paddle.sum(x.pow(2), axis=-1, keepdim=True)
        y_norm = paddle.sum(y.pow(2), axis=-1, keepdim=True)
        y_transposed = paddle.transpose(
            y, perm=[*range(y.ndim - 2), y.ndim - 1, y.ndim - 2]
        )
        y_norm_transposed = paddle.transpose(
            y_norm,
            perm=[*range(y_norm.ndim - 2), y_norm.ndim - 1, y_norm.ndim - 2],
        )
        res = paddle.matmul(x, y_transposed) * -2 + y_norm_transposed + x_norm
        res = paddle.clip(res, min=0.0).sqrt()
        return res

    return paddle.linalg.norm(
        x[..., None, :] - y[..., None, :, :], p=p, axis=-1
    )
