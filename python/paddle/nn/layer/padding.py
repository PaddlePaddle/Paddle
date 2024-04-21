# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from .common import _npairs
from .layers import Layer


class ZeroPad1D(Layer):
    """
    This interface is used to construct a callable object of the ``ZeroPad1D`` class.
    Pads the input tensor boundaries with zero.

    Parameters:
        padding (Tensor | List[int] | int): The padding size with data type int. If is int, use the
            same padding in all dimensions. Else [len(padding)/2] dimensions of input will be padded.
            The pad has the form (pad_left, pad_right).
        data_format (str): An string from: "NCL", "NCL". Specify the data format of the input data.
           Default is  "NCL"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - x(Tensor): The input tensor of zeropad1d operator, which is a 3-D tensor.
          The data type can be float32, float64.
        - output(Tensor): The output tensor of zeropad1d operator, which is a 3-D tensor.
          The data type is same as input x.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input_shape = (1, 2, 3)
            >>> pad = [1, 2]
            >>> data = paddle.arange(paddle.prod(paddle.to_tensor(input_shape)), dtype="float32").reshape(input_shape) + 1
            >>> my_pad = nn.ZeroPad1D(padding=pad)
            >>> result = my_pad(data)
            >>> print(result)
            Tensor(shape=[1, 2, 6], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[0., 1., 2., 3., 0., 0.],
              [0., 4., 5., 6., 0., 0.]]])
    """

    def __init__(self, padding, data_format="NCL", name=None):
        super().__init__()
        self._pad = _npairs(padding, 1)
        self._mode = 'constant'
        self._value = 0.0
        self._data_format = data_format
        self._name = name

    def forward(self, x):
        return F.pad(
            x,
            pad=self._pad,
            mode=self._mode,
            value=self._value,
            data_format=self._data_format,
            name=self._name,
        )

    def extra_repr(self):
        name_str = f', name={self._name}' if self._name else ''
        return f'padding={self._pad}, data_format={self._data_format}{name_str}'


class ZeroPad3D(Layer):
    """
    This interface is used to construct a callable object of the ``ZeroPad3D`` class.
    Pads the input tensor boundaries with zero.

    Parameters:
        padding (Tensor | List[int] | int): The padding size with data type int. If is int, use the
            same padding in all dimensions. Else [len(padding)/2] dimensions of input will be padded.
            The pad has the form (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back).
        data_format (str): An string from: "NCDHW", "NCDHW". Specify the data format of the input data.
           Default is  "NCDHW"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - x(Tensor): The input tensor of zeropad3d operator, which is a 5-D tensor.
          The data type can be float32, float64.
        - output(Tensor): The output tensor of zeropad3d operator, which is a 5-D tensor.
          The data type is same as input x.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input_shape = (1, 1, 1, 2, 3)
            >>> pad = [1, 0, 1, 2, 0, 0]
            >>> data = paddle.arange(paddle.prod(paddle.to_tensor(input_shape)), dtype="float32").reshape(input_shape) + 1
            >>> my_pad = nn.ZeroPad3D(padding=pad)
            >>> result = my_pad(data)
            >>> print(result)
            Tensor(shape=[1, 1, 1, 5, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[[0., 0., 0., 0.],
                [0., 1., 2., 3.],
                [0., 4., 5., 6.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]]]]])
    """

    def __init__(self, padding, data_format="NCDHW", name=None):
        super().__init__()
        self._pad = _npairs(padding, 3)
        self._mode = 'constant'
        self._value = 0.0
        self._data_format = data_format
        self._name = name

    def forward(self, x):
        return F.pad(
            x,
            pad=self._pad,
            mode=self._mode,
            value=self._value,
            data_format=self._data_format,
            name=self._name,
        )

    def extra_repr(self):
        name_str = f', name={self._name}' if self._name else ''
        return f'padding={self._pad}, data_format={self._data_format}{name_str}'
