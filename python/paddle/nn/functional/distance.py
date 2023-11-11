#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import _C_ops
from paddle.framework import in_dynamic_or_pir_mode

from ...base.data_feeder import check_type, check_variable_and_dtype
from ...base.layer_helper import LayerHelper

__all__ = []


def pairwise_distance(x, y, p=2.0, epsilon=1e-6, keepdim=False, name=None):
    r"""

    It computes the pairwise distance between two vectors. The
    distance is calculated by p-oreder norm:

    .. math::

        \Vert x \Vert _p = \left( \sum_{i=1}^n \vert x_i \vert ^ p \right) ^ {1/p}.

    Parameters:
        x (Tensor): Tensor, shape is :math:`[N, D]` or :math:`[D]`, where :math:`N`
            is batch size, :math:`D` is the dimension of vector. Available dtype is
            float16, float32, float64.
        y (Tensor): Tensor, shape is :math:`[N, D]` or :math:`[D]`, where :math:`N`
            is batch size, :math:`D` is the dimension of vector. Available dtype is
            float16, float32, float64.
        p (float, optional): The order of norm. Default: :math:`2.0`.
        epsilon (float, optional): Add small value to avoid division by zero.
            Default: :math:`1e-6`.
        keepdim (bool, optional): Whether to reserve the reduced dimension
            in the output Tensor. The result tensor is one dimension less than
            the result of ``|x-y|`` unless :attr:`keepdim` is True. Default: False.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`.
            Generally, no setting is required. Default: None.

    Returns:
        Tensor, the dtype is same as input tensor.

        - If :attr:`keepdim` is True, the output shape is :math:`[N, 1]` or :math:`[1]`,
          depending on whether the input has data shaped as :math:`[N, D]`.
        - If :attr:`keepdim` is False, the output shape is :math:`[N]` or :math:`[]`,
          depending on whether the input has data shaped as :math:`[N, D]`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([[1., 3.], [3., 5.]], dtype=paddle.float64)
            >>> y = paddle.to_tensor([[5., 6.], [7., 8.]], dtype=paddle.float64)
            >>> distance = paddle.nn.functional.pairwise_distance(x, y)
            >>> print(distance)
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [4.99999860, 4.99999860])
    """
    if in_dynamic_or_pir_mode():
        sub = _C_ops.subtract(x, y)
        # p_norm op has not used epsilon, so change it to the following.
        if epsilon != 0.0:
            epsilon = paddle.to_tensor([epsilon], dtype=sub.dtype)
            sub = _C_ops.add(sub, epsilon)
        return _C_ops.p_norm(sub, p, -1, 0.0, keepdim, False)

    else:
        check_type(p, 'porder', (float, int), 'PairwiseDistance')
        check_type(epsilon, 'epsilon', (float), 'PairwiseDistance')
        check_type(keepdim, 'keepdim', (bool), 'PairwiseDistance')

        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64'], 'PairwiseDistance'
        )
        check_variable_and_dtype(
            y, 'y', ['float16', 'float32', 'float64'], 'PairwiseDistance'
        )
        sub = paddle.subtract(x, y)
        if epsilon != 0.0:
            epsilon_var = sub.block.create_var(dtype=sub.dtype)
            epsilon_var = paddle.full(
                shape=[1], fill_value=epsilon, dtype=sub.dtype
            )
            sub = paddle.add(sub, epsilon_var)
        helper = LayerHelper("PairwiseDistance", name=name)
        attrs = {
            'axis': -1,
            'porder': p,
            'keepdim': keepdim,
            'epsilon': 0.0,
        }
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='p_norm', inputs={'X': sub}, outputs={'Out': out}, attrs=attrs
        )

        return out


def pdist(
    x, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary", name=None
):
    r'''
    Computes the p-norm distance between every pair of row vectors in the input.

    Args:
        x (Tensor): A tensor with shape :math:`N \times M`.
        p (float, optional): The value for the p-norm distance to calculate between each vector pair. Default: :math:`2.0`.
        compute_mode (str, optional): The mode for compute distance.

            - ``use_mm_for_euclid_dist_if_necessary`` , for p = 2.0 and (P > 25 or R > 25), it will use matrix multiplication to calculate euclid distance if possible.
            - ``use_mm_for_euclid_dist`` , for p = 2.0, it will use matrix multiplication to calculate euclid distance.
            - ``donot_use_mm_for_euclid_dist`` , it will not use matrix multiplication to calculate euclid distance.

            Default: ``use_mm_for_euclid_dist_if_necessary``.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor with shape: math:`N(N-1)/2` the dtype is same as input tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> a = paddle.randn([4, 5])
            >>> a
            Tensor(shape=[4, 5], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [[-0.33173719, -0.93648648, -0.01741328, -0.94435263,  2.22178721],
                    [-0.65466857,  0.10307083,  0.08741203, -0.91078597,  0.72589827],
                    [ 0.06907391, -0.27584535,  1.35355449, -0.69688839,  0.18408430],
                    [-0.00939178, -0.32901841, -1.06503606,  0.81856263,  0.16791444]])
            >>> pdist_out=paddle.pdist(a)
            >>> pdist_out
            Tensor(shape=[6], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [1.85331142, 2.58652687, 2.98273396, 1.61549115, 2.28762150, 2.85576940])

    '''

    x_shape = list(x.shape)
    assert len(x_shape) == 2, "The x must be 2-dimensional"
    d = paddle.cdist(x, x, p, compute_mode)
    mask = ~paddle.tril(paddle.ones(d.shape, dtype='bool'))
    return paddle.masked_select(d, mask)
