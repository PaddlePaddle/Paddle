#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unlessf required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import numpy
import warnings

from ..layer_helper import LayerHelper
from ..framework import (
    _current_expected_place,
    convert_np_dtype_to_dtype_,
    _varbase_creator,
    in_dygraph_mode,
)
from ..framework import Variable
from ..core import VarDesc
from .. import core
from .layer_function_generator import templatedoc
from ..data_feeder import (
    check_variable_and_dtype,
    check_type,
    check_dtype,
    convert_dtype,
)
from paddle.utils import deprecated

from paddle import _C_ops, _legacy_C_ops

__all__ = [
    'fill_constant_batch_size_like',
    'zeros',
]


def zeros(shape, dtype, force_cpu=False, name=None):
    """
    The OP creates a tensor of specified :attr:`shape` and :attr:`dtype`, and fills it with 0.
    Its :attr:`stop_gradient` will be set to True to stop gradient computation.

    Parameters:
        shape(tuple|list|Tensor): Shape of output Tensor, the data type of ``shape`` is int32 or int64.
        dtype (np.dtype|str): Data type of output Tensor, it supports
            bool, float16, float32, float64, int32 and int64.
        force_cpu (bool, optional): Whether force to store the output Tensor in CPU memory.
            If :attr:`force_cpu` is False, the output Tensor will be stored in running device memory.
            Default: False.
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A tensor of data type :attr:`dtype` with shape :attr:`shape` and all elements set to 0.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle
          data = fluid.layers.zeros(shape=[3, 2], dtype='float32') # [[0., 0.], [0., 0.], [0., 0.]]

          # shape is a Tensor
          shape = paddle.full(shape=[2], dtype='int32', fill_value=2)
          data1 = fluid.layers.zeros(shape=shape, dtype='int32') #[[0, 0], [0, 0]]
    """
    # TODO: remove zeros
    from paddle.tensor import fill_constant

    return fill_constant(
        value=0.0, shape=shape, dtype=dtype, force_cpu=force_cpu, name=name
    )
