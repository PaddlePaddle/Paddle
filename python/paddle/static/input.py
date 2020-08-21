# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import six

from ..fluid.data import data
from ..fluid import core, Variable
from ..fluid.framework import convert_np_dtype_to_dtype_

__all__ = ['data', 'InputSpec']


class InputSpec(object):
    """
    Define input specification of the model.

    Args:
        shape (tuple(integers)|list[integers]): List|Tuple of integers
            declaring the shape. You can set "None" or -1 at a dimension
            to indicate the dimension can be of any size. For example,
            it is useful to set changeable batch size as "None" or -1.
        dtype (np.dtype|VarType|str, optional): The type of the data. Supported
            dtype: bool, float16, float32, float64, int8, int16, int32, int64,
            uint8. Default: float32.
        name (str): The name/alias of the variable, see :ref:`api_guide_Name`
            for more details.

    Examples:
        .. code-block:: python

        from paddle.static import InputSpec

        input = InputSpec([None, 784], 'float32', 'x')
        label = InputSpec([None, 1], 'int64', 'label')
    """

    __slots__ = ['shape', 'dtype', 'name']

    def __init__(self, shape=None, dtype='float32', name=None):
        # replace `None` in shape  with -1
        self.shape = self._verify(shape)
        # convert dtype into united represention
        if dtype is not None:
            if not isinstance(dtype, core.VarDesc.VarType):
                dtype = convert_np_dtype_to_dtype_(dtype)
        self.dtype = dtype
        self.name = name

    def _create_feed_layer(self):
        return data(self.name, shape=self.shape, dtype=self.dtype)

    def __repr__(self):
        return '{}(shape={}, dtype={}, name={})'.format(
            type(self).__name__, self.shape, self.dtype, self.name)

    @classmethod
    def from_tensor(cls, tensor, name=None):
        """
        Generates a InputSpec based on the description of input tensor. 
        """
        if isinstance(tensor, (Variable, core.VarBase)):
            return cls(tensor.shape, tensor.dtype, name or tensor.name)
        else:
            raise ValueError(
                "Input `tensor` should be a Tensor, but received {}.".format(
                    type(tensor).__name__))

    @classmethod
    def from_numpy(cls, ndarray, name=None):
        """
        Generates a InputSpec based on the description of input np.ndarray. 
        """
        return cls(ndarray.shape, ndarray.dtype, name)

    def batch(self, batch_size):
        """
        Inserts `batch_size` in front of the `shape`.
        """
        if isinstance(batch_size, (list, tuple)):
            if len(batch_size) != 1:
                raise ValueError(
                    "Length of batch_size: {} shall be 1, but received {}.".
                    format(batch_size, len(batch_size)))
            batch_size = batch_size[1]
        elif not isinstance(batch_size, six.integer_types):
            raise TypeError("type(batch_size) shall be `int`, but received {}.".
                            format(type(batch_size).__name__))

        new_shape = [batch_size] + list(self.shape)
        return InputSpec(tuple(new_shape), self.dtype, self.name)

    def unbatch(self):
        """
        remove the first element of `shape`.
        """
        if len(self.shape) == 0:
            raise ValueError(
                "Not support to unbatch a InputSpec when len(shape) == 0.")

        return InputSpec(tuple(self.shape[1:]), self.dtype, self.name)

    def _verify(self, shape):
        if not isinstance(shape, (list, tuple)):
            raise TypeError(
                "Type of `shape` in InputSpec should be one of (tuple, list), but received {}.".
                format(type(shape).__name__))
        if len(shape) == 0:
            raise ValueError(
                "`shape` in InputSpec should contain at least 1 element, but received {}.".
                format(shape))

        for i, ele in enumerate(shape):
            if ele is not None:
                if not isinstance(ele, six.integer_types):
                    raise ValueError(
                        "shape[{}] should be an `int`, but received `{}`:{}.".
                        format(i, type(ele).__name__, ele))
            if ele is None or ele < -1:
                shape[i] = -1

        return tuple(shape)

    def __hash__(self):
        return hash((tuple(self.shape), self.dtype))

    def __eq__(self, other):
        return (type(self) is type(other) and all(
            getattr(self, attr) == getattr(other, attr)
            for attr in self.__slots__))

    def __ne__(self, other):
        return not self == other
