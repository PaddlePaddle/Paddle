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

# TODO: define functions of linear algebra   
__all__ = [
    'matmul',
    #  'dot',
    #  'einsum',
    #  'morm',
    #  'transpose',
'dist',
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


def dist(x, y, p=2):
    """
    This OP returns the p-norm of (x - y). The shapes of x and y must be broadcastable.
    where, z = x - y,

    .. math::

        \left \| z \right \|_{p} = (\sum_{i=i}^{m} |z_i|^p)^{1/p}

    when p = 0, the 0-norm of z is simply the number of non-zero elements of z.
    when p = inf, the inf-norm of z is the maximum element of z.
    when p = -inf, the inf-norm of z is the minimum element of z.

    Args:
        x (Variable): The input Tensor, its data type is float32 and float64.
        y (Variable): The Right-hand-side input Tensor, its data type is float32 or float64.
        p (float, optional): The norm to be computed, its data type is float32 or float64. Default: 2.

    Returns:
        Variable: Tensor that is the p-norm of (x - y).

    Raises:
        TypeError: If the data type of `x` or `y` is not float32 or float64.
        TypeError: If the type of `p` is not float or int.

    Examples:
        .. code-block:: python
            import paddle
            import paddle.fluid as fluid

            is_use_gpu = False
            x = fluid.layers.fill_constant(shape=[2, 2], value=3, dtype='float32')
            y = fluid.layers.fill_constant(shape=[2, 2], value=1, dtype='float32')
            out = paddle.dist(x, y, 2) # out = [8]

            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            exe = fluid.Executor(place)
            result = exe.run(fluid.default_main_program(),
                          fetch_list=[result])
            print(result[0])

    """
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
