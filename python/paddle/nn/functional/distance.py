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
    distance is calculated by p-order norm:

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


def pdist(x, p=2.0, name=None):
    r'''
    Computes the p-norm distance between every pair of row vectors in the input.

    Args:
        x (Tensor): The input tensor with shape :math:`N \times M`.
        p (float, optional): The value for the p-norm distance to calculate between each vector pair. Default: :math:`2.0`.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor with shape :math:`N(N-1)/2` , the dtype is same as input tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> a = paddle.randn([4, 5])
            >>> print(a)
            Tensor(shape=[4, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[ 0.06132207,  1.11349595,  0.41906244, -0.24858207, -1.85169315],
                    [-1.50370061,  1.73954511,  0.13331604,  1.66359663, -0.55764782],
                    [-0.59911072, -0.57773495, -1.03176904, -0.33741450, -0.29695082],
                    [-1.50258386,  0.67233968, -1.07747352,  0.80170447, -0.06695852]])
            >>> pdist_out=paddle.pdist(a)
            >>> print(pdist_out)
            Tensor(shape=[6], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [2.87295413, 2.79758120, 3.02793980, 3.40844536, 1.89435327, 1.93171620])
    '''

    x_shape = list(x.shape)
    assert len(x_shape) == 2, "The x must be 2-dimensional"
    d = paddle.linalg.norm(x[..., None, :] - x[..., None, :, :], p=p, axis=-1)
    mask = ~paddle.tril(paddle.ones(d.shape, dtype='bool'))
    return paddle.masked_select(d, mask)
