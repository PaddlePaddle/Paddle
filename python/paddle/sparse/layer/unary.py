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

from .. import functional as F
from paddle.nn import Layer

__all__ = []


class ReLU(Layer):
    """
    Sparse ReLU Activation.

    .. math::

        ReLU(x) = max(x, 0)

    Parameters:
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Sparse Tensor with any shape.
        - output: Sparse Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.fluid.framework import _test_eager_guard
            with _test_eager_guard():
                x = [[0, -1, 0, 2], [0, 0, -3, 0], [4, 5, 0, 0]]
                dense_x = paddle.to_tensor(x, dtype='float32')
                sparse_dim = 2
                sparse_x = dense_x.to_sparse_coo(sparse_dim)
                relu = paddle.sparse.ReLU()
                out = relu(sparse_x)
                #out.values: [0., 2., 0., 4., 5.]
    """

    def __init__(self, name=None):
        super(ReLU, self).__init__()
        self._name = name

    def forward(self, x):
        return F.relu(x, self._name)

    def extra_repr(self):
        name_str = 'name={}'.format(self._name) if self._name else ''
        return name_str
