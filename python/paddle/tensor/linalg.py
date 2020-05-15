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
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type
from ..fluid.framework import in_dygraph_mode, _varbase_creator

from ..fluid.layers import transpose  #DEFINE_ALIAS

__all__ = [
    'matmul',
    'dot',
    #       'einsum',
    'norm',
    'transpose',
    'dist',
    't',
    'cross',
    'cholesky',
    #       'tensordot',
    'bmm'
]


def matmul(x, y, transpose_x=False, transpose_y=False, alpha=1.0, name=None):
    """
	:alias_main: paddle.matmul
	:alias: paddle.matmul,paddle.tensor.matmul,paddle.tensor.linalg.matmul

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

            import paddle
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
        out = _varbase_creator(dtype=x.dtype)
        core.ops.matmul(x, y, out, 'transpose_X', transpose_x, 'transpose_Y',
                        transpose_y, 'alpha', float(alpha))
        return out

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
	:alias_main: paddle.norm
	:alias: paddle.norm,paddle.tensor.norm,paddle.tensor.linalg.norm

    Returns the matrix norm (Frobenius) or vector norm (the 1-norm, the Euclidean
    or 2-norm, and in general the p-norm for p > 0) of a given tensor.

    Args:
        input (Variable): The input tensor could be N-D tensor, and the input data
            type could be float32 or float64.
        p (float|string, optional): Order of the norm. Supported values are `fro`, `1`, `2`,
            and any positive real number yielding the corresponding p-norm.
        axis (int|list, optional): The axis on which to apply norm operation. If axis is int
            or list with only one element, the vector norm is computed over the axis.
            If axis is a list with two elements, the matrix norm is computed over the axis.
            If `axis < 0`, the dimension to norm operation is rank(input) + axis.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have fewer dimension
            than the :attr:`input` unless :attr:`keepdim` is true, default
            value is False.
        out (Variable, optional): The output tensor, default value is None. It's data type
            must be the same as the input Tensor.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Variable: Tensor, results of norm operation on the specified axis of input tensor,
        it's data type is the same as input's Tensor.
 
    Raises:
        TypeError, if out data type is different with the input data type.
        ValueError, If `p` or `axis` is invalid.
    
    Examples:
        .. code-block:: python
            
            import paddle
            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[2, 3, 5], dtype='float64')
            
            # compute frobenius norm along last two dimensions.
            out_fro = paddle.norm(x, p='fro', axis=[1,2])
            
            # compute 2-order vector norm along last dimension.
            out_pnorm = paddle.norm(x, p=2, axis=-1)
    """

    def frobenius_norm(input, dim=None, keepdim=False, out=None, name=None):
        """
        The frobenius norm OP is to calculate the frobenius norm of certain two dimensions of Tensor `input`.
        Args:
          input (Variable): Tensor, data type float32, float64.
          dim (list, optional): None for last two dimensions.
          keepdim (bool, optional): Whether keep the dimensions as the `input`, Default False.
          out (Variable, optional): The tensor variable storing the output.
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
        if len(attrs['dim']) == len(input.shape):
            attrs['reduce_all'] = True
        check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                                 'frobenius_norm')

        helper = LayerHelper('frobenius_norm', **locals())
        if out is None:
            out = helper.create_variable_for_type_inference(
                dtype=helper.input_dtype())
        else:
            check_type(out, 'out', (Variable), 'frobenius_norm')
            check_dtype(
                out.dtype, out.name,
                convert_dtype(input.dtype), 'frobenius_norm',
                '(The out data type in frobenius_norm must be the same with input data type.)'
            )

        helper.append_op(
            type='frobenius_norm',
            inputs={'X': input},
            outputs={'Out': out},
            attrs=attrs)
        return out

    def vector_norm(input,
                    porder=None,
                    axis=None,
                    keepdim=False,
                    out=None,
                    name=None):
        """
        Calculate the p-order vector norm for certain  dimension of Tensor `input`.
        Args:
          input (Variable): Tensor, data type float32, float64.
          porder (float, optional): None for porder=2.0.
          axis (int, optional): None for last dimension.
          keepdim (bool, optional): Whether keep the dimensions as the `input`, Default False.
          out (Variable, optional): The tensor variable storing the output.
        """
        if porder is not None:
            check_type(porder, 'porder', (float, int), 'p_norm')
        if axis is not None:
            check_type(axis, 'axis', (int), 'p_norm')
        attrs = {
            'axis': axis if axis is not None else -1,
            'porder': float(porder) if porder is not None else 2.0,
            'keepdim': keepdim,
            'epsilon': 1e-12,
        }
        check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                                 'p_norm')

        helper = LayerHelper('p_norm', **locals())
        if out is None:
            out = helper.create_variable_for_type_inference(
                dtype=helper.input_dtype())
        else:
            check_type(out, 'out', (Variable), 'p_norm')
            check_dtype(
                out.dtype, out.name,
                convert_dtype(input.dtype), 'p_norm',
                '(The out data type in p_norm must be the same with input data type.)'
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
                    input, dim=axis, keepdim=keepdim, out=out, name=name)
            else:
                raise ValueError(
                    "only valid string values are 'fro', found {}".format(p))
        elif isinstance(p, (int, float)):
            return vector_norm(
                input, porder=p, axis=axis, keepdim=keepdim, out=out, name=name)
        else:
            raise ValueError("only valid p type is string or float, found {}".
                             format(type(p)))

    if isinstance(axis, list) and len(axis) == 1:
        axis = axis[0]

    #calculate vector norm, where axis is int or list with only one integer
    if isinstance(axis, int):
        if isinstance(p, (int, float)):
            return vector_norm(
                input, axis=axis, porder=p, keepdim=keepdim, out=out, name=name)
        else:
            raise ValueError(
                "unspport p for p-order vector norm. except float, found {}".
                format(p))
    #calculate matrix norm, where axis is list with two integers
    elif isinstance(axis, list) and len(axis) == 2:
        if p == "fro":
            return frobenius_norm(
                input, dim=axis, keepdim=keepdim, out=out, name=name)
        else:
            raise ValueError(
                "unspport p for matrix norm, expcept 'fro', found {}".format(p))
    else:
        raise ValueError(
            "except axis type int or list (length of list <=2), found {}".
            format(axis))


def dist(x, y, p=2):
    """
	:alias_main: paddle.dist
	:alias: paddle.dist,paddle.tensor.dist,paddle.tensor.linalg.dist

    This OP returns the p-norm of (x - y). It is not a norm in a strict sense, only as a measure
    of distance. The shapes of x and y must be broadcastable. The definition is as follows, for
    details, please refer to the `numpy's broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_:

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

    When p = inf, the inf-norm of z is the maximum element of z.

    .. math::

        ||z||_\infty=\max_i |z_i|

    When p = -inf, the negative-inf-norm of z is the minimum element of z.

    .. math::

        ||z||_{-\infty}=\min_i |z_i|

    Otherwise, the p-norm of z follows the formula,

    .. math::

        ||z||_{p}=(\sum_{i=1}^{m}|z_i|^p)^{\\frac{1}{p}}

    Args:
        x (Variable): 1-D to 6-D Tensor, its data type is float32 or float64.
        y (Variable): 1-D to 6-D Tensor, its data type is float32 or float64.
        p (float, optional): The norm to be computed, its data type is float32 or float64. Default: 2.

    Returns:
        Variable: Tensor that is the p-norm of (x - y).

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            with fluid.dygraph.guard():
                x = fluid.dygraph.to_variable(np.array([[3, 3],[3, 3]]).astype(np.float32))
                y = fluid.dygraph.to_variable(np.array([[3, 3],[3, 1]]).astype(np.float32))
                out = paddle.dist(x, y, 0)
                print(out.numpy()) # out = [1.]

                out = paddle.dist(x, y, 2)
                print(out.numpy()) # out = [2.]

                out = paddle.dist(x, y, float("inf"))
                print(out.numpy()) # out = [2.]

                out = paddle.dist(x, y, float("-inf"))
                print(out.numpy()) # out = [0.]
    """
    check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'dist')
    check_variable_and_dtype(y, 'dtype', ['float32', 'float64'], 'dist')
    check_type(p, 'p', (float, int), 'dist')
    helper = LayerHelper("dist", **locals())
    out = helper.create_variable_for_type_inference(x.dtype)

    inputs = {"X": [x], "Y": [y]}
    outputs = {'Out': [out]}
    attrs = {"p": float(p)}
    helper.append_op(
        type='dist', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


def dot(x, y, name=None):
    """
	:alias_main: paddle.dot
	:alias: paddle.dot,paddle.tensor.dot,paddle.tensor.linalg.dot

    This operator calculates inner product for vectors.
   
    .. note::
       Only support 1-d Tensor(vector).

    Parameters:
        x(Variable): 1-D ``Tensor`` or ``LoDTensor``. Its datatype should be ``float32``, ``float64``, ``int32``, ``int64``
        y(Variable): 1-D ``Tensor`` or ``LoDTensor``. Its datatype soulde be ``float32``, ``float64``, ``int32``, ``int64``
        name(str, optional): Name of the output. Default is None. It's used to print debug info for developers. Details: :ref:`api_guide_Name`

    Returns:
        Variable: the calculated result Tensor/LoDTensor.

    Examples:

    .. code-block:: python

        import paddle
        import paddle.fluid as fluid
        import numpy as np
        
        with fluid.dygraph.guard():
          x = fluid.dygraph.to_variable(np.random.uniform(0.1, 1, [10]).astype(np.float32))
          y = fluid.dygraph.to_variable(np.random.uniform(1, 3, [10]).astype(np.float32))
          z = paddle.dot(x, y)
          print(z.numpy())

    """
    op_type = 'dot'
    # skip var type check in dygraph mode to improve efficiency
    if in_dygraph_mode():
        op = getattr(core.ops, op_type)
        return op(x, y)

    assert x is not None, 'x cannot be None in {}'.format(op_type)
    assert y is not None, 'y cannot be None in {}'.format(op_type)

    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             op_type)
    check_variable_and_dtype(y, 'y', ['float32', 'float64', 'int32', 'int64'],
                             op_type)

    helper = LayerHelper(op_type, **locals())
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)
    helper.append_op(
        type="dot", inputs={'X': x,
                            'Y': y}, attrs={}, outputs={"Out": out})
    return out


def t(input, name=None):
    """
	:alias_main: paddle.t
	:alias: paddle.t,paddle.tensor.t,paddle.tensor.linalg.t

    Transpose <=2-D tensor. 
    0-D and 1-D tensors are returned as it is and 2-D tensor is equal to 
    the fluid.layers.transpose function which perm dimensions set 0 and 1.
    
    Args:
        input (Variable): The input Tensor. It is a N-D (N<=2) Tensor of data types float16, float32, float64, int32.
        name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`
    Returns:
        Variable: A transposed n-D Tensor, with data type being float16, float32, float64, int32, int64.
    
    For Example:
        .. code-block:: text
        # Example 1 (0-D tensor)
         x = tensor([0.79])
         paddle.t(x) = tensor([0.79])
         # Example 2 (1-D tensor)
         x = tensor([0.79, 0.84, 0.32])
         paddle.t(x) = tensor([0.79, 0.84, 0.32])
        
         # Example 3 (2-D tensor)
         x = tensor([0.79, 0.84, 0.32],
                    [0.64, 0.14, 0.57])
         paddle.t(x) = tensor([0.79, 0.64],
                              [0.84, 0.14],
                              [0.32, 0.57])
    
     Examples:
        .. code-block:: python
            import paddle
            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[2, 3],
                            dtype='float32')
            x_transposed = paddle.t(x)
            print x_transposed.shape
            #(3L, 2L)
    """
    if len(input.shape) > 2:
        raise ValueError(
            "Input(input) only support N-D (N<=2) tensor, but received "
            "length of Input(input) is %s. Perhaps you can use paddle."
            "tensor.transpose() instead." % len(input.shape))
    if in_dygraph_mode():
        if len(input.shape) == 1:
            return input
        # 2-D tensor
        perm = [1, 0]
        out, _ = core.ops.transpose2(input, 'axis', perm)
        return out

    check_variable_and_dtype(
        input, 'input', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'transpose')

    helper = LayerHelper('t', **locals())
    out = helper.create_variable_for_type_inference(input.dtype)
    input_shape = helper.create_variable_for_type_inference(input.dtype)
    if len(input.shape) == 1:
        out = input
    else:
        helper.append_op(
            type='transpose2',
            inputs={'X': [input]},
            outputs={'Out': [out],
                     'XShape': [input_shape]},
            attrs={'axis': [1, 0]})
    return out


def cross(input, other, dim=None):
    """
	:alias_main: paddle.cross
	:alias: paddle.cross,paddle.tensor.cross,paddle.tensor.linalg.cross

    Returns the cross product of vectors in dimension `dim` of the `input` and `other` tensor. 
    Inputs must have the same shape, and the size of their dim-th dimension should be equla to 3. 
    If `dim` is not given, it defaults to the first dimension found with the size 3.
    
    Args:
        input (Variable): The first input tensor variable.
        other (Variable): The second input tensor variable.
        dim (int): The dimension to take the cross-product in.

    Returns:
        Variable: A Tensor with same data type as `input`.
        
    Examples:
        .. code-block:: python
            import paddle
            import paddle.fluid as fluid
            import numpy as np

            data_x = np.array([[1.0, 1.0, 1.0],
                               [2.0, 2.0, 2.0],
                               [3.0, 3.0, 3.0]])
            data_y = np.array([[1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0]])

            with fluid.dygraph.guard():
                x = fluid.dygraph.to_variable(data_x)
                y = fluid.dygraph.to_variable(data_y)
                out_z1 = paddle.cross(x, y)
                print(out_z1.numpy())
                #[[-1. -1. -1.]
                # [ 2.  2.  2.]
                # [-1. -1. -1.]]
                out_z2 = paddle.cross(x, y, dim=1)
                print(out_z2.numpy())
                #[[0. 0. 0.]
                # [0. 0. 0.]
                # [0. 0. 0.]]
    """
    helper = LayerHelper("cross", **locals())
    if in_dygraph_mode():
        if dim:
            return core.ops.cross(input, other, 'dim', dim)
        else:
            return core.ops.cross(input, other)

    out = helper.create_variable_for_type_inference(input.dtype)
    attrs = dict()
    if dim:
        attrs['dim'] = dim

    helper.append_op(
        type='cross',
        inputs={'X': input,
                'Y': other},
        outputs={'Out': out},
        attrs=attrs)
    return out


def cholesky(x, upper=False):
    """
	:alias_main: paddle.cholesky
	:alias: paddle.cholesky,paddle.tensor.cholesky,paddle.tensor.linalg.cholesky

    Computes the Cholesky decomposition of one symmetric positive-definite
    matrix or batches of symmetric positive-definite matrice. 
    
    If `upper` is `True`, the decomposition has the form :math:`A = U^{T}U` ,
    and the returned matrix :math:`U` is upper-triangular. Otherwise, the
    decomposition has the form  :math:`A = LL^{T}` , and the returned matrix
    :math:`L` is lower-triangular.

    Args:
        x (Variable): The input tensor. Its shape should be `[*, M, M]`,
            where * is zero or more batch dimensions, and matrices on the
            inner-most 2 dimensions all should be symmetric positive-definite.
            Its data type should be float32 or float64.
        upper (bool): The flag indicating whether to return upper or lower
            triangular matrices. Default: False.

    Returns:
        Variable: A Tensor with same shape and data type as `x`. It represents \
            triangular matrices generated by Cholesky decomposition.
        
    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            with fluid.dygraph.guard():
                a = np.random.rand(3, 3)
                a_t = np.transpose(a, [1, 0])
                x = np.matmul(a, a_t) + 1e-03
                x = fluid.dygraph.to_variable(x)
                out = paddle.cholesky(x, upper=False)
                print(out.numpy())
                # [[1.190523   0.         0.        ]
                #  [0.9906703  0.27676893 0.        ]
                #  [1.25450498 0.05600871 0.06400121]]

    """
    check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'cholesky')
    check_type(upper, 'upper', bool, 'cholesky')
    helper = LayerHelper('cholesky', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='cholesky',
        inputs={'X': [x]},
        outputs={'Out': out},
        attrs={'upper': upper})
    return out


def bmm(x, y, name=None):
    """
	:alias_main: paddle.bmm
	:alias: paddle.bmm,paddle.tensor.bmm,paddle.tensor.linalg.bmm

    Applies batched matrix multiplication to two tensors.

    Both of the two input tensors must be three-dementional and share the same batch size.

    if x is a (b, m, k) tensor, y is a (b, k, n) tensor, the output will be a (b, m, n) tensor.

    Args:
        x (Variable): The input variable which is a Tensor or LoDTensor.
        y (Variable): The input variable which is a Tensor or LoDTensor.
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Variable: The product Tensor (or LoDTensor) variable.

    Examples:
        import paddle
        import paddle.fluid as fluid
        x = fluid.layers.data(name='x', shape=[10, 3, 4], dtype='float32')
        y = fluid.layers.data(name='y', shape=[10, 4, 5], dtype='float32')
        out = paddle.bmm(x, y)
    
        # In dygraph mode:
        # size input1: (2, 2, 3) and input2: (2, 3, 2)
        input1 = np.array([[[1.0, 1.0, 1.0],[2.0, 2.0, 2.0]],[[3.0, 3.0, 3.0],[4.0, 4.0, 4.0]]])
        input2 = np.array([[[1.0, 1.0],[2.0, 2.0],[3.0, 3.0]],[[4.0, 4.0],[5.0, 5.0],[6.0, 6.0]]])

        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(input1)
            y = fluid.dygraph.to_variable(input2)
            out = paddle.bmm(x, y)
            #output size: (2, 2, 2)
            #output value:
            #[[[6.0, 6.0],[12.0, 12.0]],[[45.0, 45.0],[60.0, 60.0]]]
            out_np = out.numpy()
    """

    helper = LayerHelper('bmm', **locals())
    if in_dygraph_mode():
        return core.ops.bmm(x, y)
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='bmm', inputs={'X': x, 'Y': y}, outputs={'Out': out})
    return out
