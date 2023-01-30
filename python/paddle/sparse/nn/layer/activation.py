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

<<<<<<< HEAD
from paddle.nn import Layer

from .. import functional as F
=======
from .. import functional as F
from paddle.nn import Layer
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

__all__ = []


class ReLU(Layer):
    """

    Sparse ReLU Activation, requiring x to be a SparseCooTensor or SparseCsrTensor.

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

            dense_x = paddle.to_tensor([-2., 0., 1.])
            sparse_x = dense_x.to_sparse_coo(1)
            relu = paddle.sparse.nn.ReLU()
            out = relu(sparse_x)
            # [0., 0., 1.]

    """

    def __init__(self, name=None):
<<<<<<< HEAD
        super().__init__()
=======
        super(ReLU, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._name = name

    def forward(self, x):
        return F.relu(x, self._name)

    def extra_repr(self):
        name_str = 'name={}'.format(self._name) if self._name else ''
        return name_str


class Softmax(Layer):
    r"""

    Sparse Softmax Activation, requiring x to be a SparseCooTensor or SparseCsrTensor.

    Note:
        Only support axis=-1 for SparseCsrTensor, which is faster when read data
        by row (axis=-1).

<<<<<<< HEAD
    Transform x to dense matix, and :math:`i` is row index, :math:`j` is column index.
    If axis=-1, We have:
=======
    From the point of view of dense matrix, for each row :math:`i` and each column :math:`j`
    in the matrix, we have:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    .. math::

        softmax_ij = \frac{\exp(x_ij - max_j(x_ij))}{\sum_j(exp(x_ij - max_j(x_ij))}

    Parameters:
        axis (int, optional): The axis along which to perform softmax calculations. Only support -1 for SparseCsrTensor.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: SparseCooTensor / SparseCsrTensor with any shape.
        - output: Sparse Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle
<<<<<<< HEAD
            paddle.seed(2022)

            mask = paddle.rand((3, 4)) < 0.7
            x = paddle.rand((3, 4)) * mask
            print(x)
            # Tensor(shape=[3, 4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [[0.08325022, 0.27030438, 0.        , 0.83883715],
            #         [0.        , 0.95856029, 0.24004589, 0.        ],
            #         [0.14500992, 0.17088132, 0.        , 0.        ]])

            csr = x.to_sparse_csr()
            print(csr)
            # Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
            #        crows=[0, 3, 5, 7],
            #        cols=[0, 1, 3, 1, 2, 0, 1],
            #        values=[0.08325022, 0.27030438, 0.83883715, 0.95856029, 0.24004589,
            #                0.14500992, 0.17088132])

            softmax = paddle.sparse.nn.Softmax()
            out = softmax(csr)
            print(out)
            # Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
            #        crows=[0, 3, 5, 7],
            #        cols=[0, 1, 3, 1, 2, 0, 1],
            #        values=[0.23070428, 0.27815846, 0.49113727, 0.67227983, 0.32772022,
            #                0.49353254, 0.50646752])
    """

    def __init__(self, axis=-1, name=None):
        super().__init__()
=======
            import numpy as np
            paddle.seed(100)

            mask = np.random.rand(3, 4) < 0.5
            np_x = np.random.rand(3, 4) * mask
            # [[0.         0.         0.96823406 0.19722934]
            #  [0.94373937 0.         0.02060066 0.71456372]
            #  [0.         0.         0.         0.98275049]]

            csr = paddle.to_tensor(np_x).to_sparse_csr()
            # Tensor(shape=[3, 4], dtype=paddle.float64, place=Place(gpu:0), stop_gradient=True,
            #        crows=[0, 2, 5, 6],
            #        cols=[2, 3, 0, 2, 3, 3],
            #        values=[0.96823406, 0.19722934, 0.94373937, 0.02060066, 0.71456372,
            #                0.98275049])

            softmax = paddle.sparse.nn.Softmax()
            out = softmax(csr)
            # Tensor(shape=[3, 4], dtype=paddle.float64, place=Place(gpu:0), stop_gradient=True,
            #        crows=[0, 2, 5, 6],
            #        cols=[2, 3, 0, 2, 3, 3],
            #        values=[0.68373820, 0.31626180, 0.45610887, 0.18119845, 0.36269269,
            #                1.        ])
    """

    def __init__(self, axis=-1, name=None):
        super(Softmax, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._axis = axis
        self._name = name

    def forward(self, x):
        return F.softmax(x, self._axis, self._name)

    def extra_repr(self):
        name_str = 'name={}'.format(self._name) if self._name else ''
        return name_str


class ReLU6(Layer):
    """

    Sparse ReLU6 Activation, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

<<<<<<< HEAD
        ReLU6(x) = min(max(0,x), 6)
=======
        ReLU(x) = min(max(0,x), 6)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    Parameters:
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Sparse Tensor with any shape.
        - output: Sparse Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([-2., 0., 8.])
            sparse_x = dense_x.to_sparse_coo(1)
            relu6 = paddle.sparse.nn.ReLU6()
            out = relu6(sparse_x)

    """

    def __init__(self, name=None):
<<<<<<< HEAD
        super().__init__()
=======
        super(ReLU6, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._name = name

    def forward(self, x):
        return F.relu6(x, self._name)

    def extra_repr(self):
        name_str = 'name={}'.format(self._name) if self._name else ''
        return name_str


class LeakyReLU(Layer):
    r"""

    Sparse Leaky ReLU Activation, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        LeakyReLU(x)=
            \left\{
                \begin{array}{rcl}
                    x, & & if \ x >= 0 \\
                    negative\_slope * x, & & otherwise \\
                \end{array}
            \right.

    Parameters:
        negative_slope (float, optional): Slope of the activation function at
            :math:`x < 0` . Default is 0.01.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Sparse Tensor with any shape.
        - output: Sparse Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([-2., 0., 5.])
            sparse_x = dense_x.to_sparse_coo(1)
            leaky_relu = paddle.sparse.nn.LeakyReLU(0.5)
            out = leaky_relu(sparse_x)

    """

    def __init__(self, negative_slope=0.01, name=None):
<<<<<<< HEAD
        super().__init__()
=======
        super(LeakyReLU, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._negative_slope = negative_slope
        self._name = name

    def forward(self, x):
        return F.leaky_relu(x, self._negative_slope, self._name)

    def extra_repr(self):
        name_str = 'name={}'.format(self._name) if self._name else ''
        return name_str
