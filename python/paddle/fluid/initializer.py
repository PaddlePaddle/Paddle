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

import math
import functools
from . import framework
from . import core
from .framework import (
    in_dygraph_mode,
    default_main_program,
    _current_expected_place,
)
from .lazy_init import lazy_init_helper
from .framework import program_guard
import numpy as np
from .core import VarDesc
from . import unique_name
from .data_feeder import check_variable_and_dtype, check_type, check_dtype
from paddle import _C_ops, _legacy_C_ops
import paddle

__all__ = []

_global_weight_initializer_ = None
_global_bias_initializer_ = None


def _global_weight_initializer():
    """
    Return the global weight initializer, The user doesn't need to use it.
    """
    return _global_weight_initializer_


def _global_bias_initializer():
    """
    Return the global weight initializer, The user doesn't need to use it.
    """
    return _global_bias_initializer_


def calculate_gain(nonlinearity, param=None):
    """
    Get the recommended ``gain`` value of some nonlinearity function. ``gain`` value can be used in some
    ``paddle.nn.initializer`` api to adjust the initialization value.

    Args:
        nonlinearity(str): name of nonlinearity activation function. If it is a linear function, such as:
            `linear/conv1d/conv2d/conv3d/conv1d_transpose/conv2d_transpose/conv3d_transpose` , 1.0 will be returned.
        param(bool|int|float, optional): optional parameter for somme nonlinearity function. Now, it only applies to
            'leaky_relu'. Default: None, it will be calculated as 0.01 in the formula.

    Returns:
        A float value, which is the recommended gain for this nonlinearity function.

    Examples:
        .. code-block:: python

            import paddle
            gain = paddle.nn.initializer.calculate_gain('tanh') # 5.0 / 3
            gain = paddle.nn.initializer.calculate_gain('leaky_relu', param=1.0) # 1.0 = math.sqrt(2.0 / (1+param^2))
            initializer = paddle.nn.initializer.Orthogonal(gain)

    """
    if param is None:
        param = 0.01
    else:
        assert isinstance(param, (bool, int, float))
        param = float(param)
    recommended_gain = {
        'sigmoid': 1,
        'linear': 1,
        'conv1d': 1,
        'conv2d': 1,
        'conv3d': 1,
        'conv1d_transpose': 1,
        'conv2d_transpose': 1,
        'conv3d_transpose': 1,
        'tanh': 5.0 / 3,
        'relu': math.sqrt(2.0),
        'leaky_relu': math.sqrt(2.0 / (1 + param**2)),
        'selu': 3.0 / 4,
    }
    if nonlinearity in recommended_gain.keys():
        return recommended_gain[nonlinearity]
    else:
        raise ValueError(
            "nonlinearity function {} is not suppported now.".format(
                nonlinearity
            )
        )
