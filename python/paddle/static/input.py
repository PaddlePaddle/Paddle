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

from . import data


class InputSpec(object):
    """
    Define input specification of the model.

    Args:
        name (str): The name/alias of the variable, see :ref:`api_guide_Name`
            for more details.
        shape (tuple(integers)|list[integers]): List|Tuple of integers
            declaring the shape. You can set "None" or -1 at a dimension
            to indicate the dimension can be of any size. For example,
            it is useful to set changeable batch size as "None" or -1.
        dtype (np.dtype|VarType|str, optional): The type of the data. Supported
            dtype: bool, float16, float32, float64, int8, int16, int32, int64,
            uint8. Default: float32.

    Examples:
        .. code-block:: python

        from paddle.static import InputSpec

        input = InputSpec([None, 784], 'float32', 'x')
        label = InputSpec([None, 1], 'int64', 'label')
    """

    def __init__(self, shape=None, dtype='float32', name=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name

    def _create_feed_layer(self):
        return data(self.name, shape=self.shape, dtype=self.dtype)
