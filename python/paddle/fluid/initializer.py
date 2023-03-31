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

__all__ = ['set_global_initializer']

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


def set_global_initializer(weight_init, bias_init=None):
    """
    This API is used to set up global model parameter initializer in framework.

    After this API is invoked, the global initializer will takes effect in subsequent code.

    The model parameters include ``weight`` and ``bias`` . In the framework, they correspond
    to ``paddle.ParamAttr`` , which is inherited from ``paddle.Tensor`` , and is a persistable Variable.
    This API only takes effect for model parameters, not for variables created through apis such as
    :ref:`api_fluid_layers_create_global_var` , :ref:`api_fluid_layers_create_tensor`.

    If the initializer is also set up by ``param_attr`` or ``bias_attr`` when creating a network layer,
    the global initializer setting here will not take effect because it has a lower priority.

    If you want to cancel the global initializer in framework, please set global initializer to ``None`` .

    Args:
        weight_init (Initializer): set the global initializer for ``weight`` of model parameters.
        bias_init (Initializer, optional): set the global initializer for ``bias`` of model parameters.
            Default: None.

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn

            nn.initializer.set_global_initializer(nn.initializer.Uniform(), nn.initializer.Constant())
            x_var = paddle.uniform((2, 4, 8, 8), dtype='float32', min=-1., max=1.)

            # The weight of conv1 is initialized by Uniform
            # The bias of conv1 is initialized by Constant
            conv1 = nn.Conv2D(4, 6, (3, 3))
            y_var1 = conv1(x_var)

            # If set param_attr/bias_attr too, global initializer will not take effect
            # The weight of conv2 is initialized by Xavier
            # The bias of conv2 is initialized by Normal
            conv2 = nn.Conv2D(4, 6, (3, 3),
                weight_attr=nn.initializer.XavierUniform(),
                bias_attr=nn.initializer.Normal())
            y_var2 = conv2(x_var)

            # Cancel the global initializer in framework, it will takes effect in subsequent code
            nn.initializer.set_global_initializer(None)
    """

    check_type(
        weight_init,
        'weight_init',
        (paddle.nn.initializer.Initializer, type(None)),
        'set_global_initializer',
    )
    global _global_weight_initializer_
    _global_weight_initializer_ = weight_init

    check_type(
        bias_init,
        'bias_init',
        (paddle.nn.initializer.Initializer, type(None)),
        'set_global_initializer',
    )
    global _global_bias_initializer_
    _global_bias_initializer_ = bias_init
