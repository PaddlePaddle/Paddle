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

import numpy as np
import paddle
from .. import Layer
from .. import functional as F

__all__ = []


class PairwiseDistance(Layer):
    r"""
    This operator computes the pairwise distance between two vectors. The
    distance is calculated by p-oreder norm:

    .. math::

        \Vert x \Vert _p = \left( \sum_{i=1}^n \vert x_i \vert ^ p \right) ^ {1/p}.

    Parameters:
        p (float): The order of norm. The default value is 2.
        epsilon (float, optional): Add small value to avoid division by zero,
            default value is 1e-6.
        keepdim (bool, optional): Whether to reserve the reduced dimension
            in the output Tensor. The result tensor is one dimension less than
            the result of ``'x-y'`` unless :attr:`keepdim` is True, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        x: :math:`[N, D]` where `D` is the dimension of vector, available dtype
            is float32, float64.
        y: :math:`[N, D]`, y have the same shape and dtype as x.
        out: :math:`[N]`. If :attr:`keepdim` is ``True``, the out shape is :math:`[N, 1]`.
            The same dtype as input tensor.

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.to_tensor([[1., 3.], [3., 5.]], dtype=paddle.float64)
            y = paddle.to_tensor([[5., 6.], [7., 8.]], dtype=paddle.float64)
            dist = paddle.nn.PairwiseDistance()
            distance = dist(x, y)
            print(distance.numpy()) # [5. 5.]

    """

    def __init__(self, p=2., epsilon=1e-6, keepdim=False, name=None):
        super(PairwiseDistance, self).__init__()
        self.p = p
        self.epsilon = epsilon
        self.keepdim = keepdim
        self.name = name

    def forward(self, x, y):

        return F.pairwise_distance(x, y, self.p, self.epsilon, self.keepdim,
                                   self.name)

    def extra_repr(self):
        main_str = 'p={p}'
        if self.epsilon != 1e-6:
            main_str += ', epsilon={epsilon}'
        if self.keepdim != False:
            main_str += ', keepdim={keepdim}'
        if self.name != None:
            main_str += ', name={name}'
        return main_str.format(**self.__dict__)
