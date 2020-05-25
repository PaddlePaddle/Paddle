#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
from ...wrapped_decorator import signature_safe_contextmanager, wrap_decorator
from paddle.fluid import core
import contextlib
from ...framework import Variable, in_dygraph_mode, OpProtoHolder, Parameter, _dygraph_tracer, dygraph_only
import warnings
import copy

__all__ = ['autocast']

# The set of ops that support fp16 calculation and are considered numerically-
# safe and performance-critical. These ops are always converted to fp16.
white_list = {
    'conv2d',
    'matmul',
    'mul',
}

# The set of ops that support fp16 calculation and are considered numerically-
# dangerous and whose effects may also be observed in downstream ops.
black_list = {
    'exp',
    'square',
    'log',
    'mean',
    'sum',
    'cos_sim',
    'softmax',
    'softmax_with_cross_entropy',
    'sigmoid_cross_entropy_with_logits',
    'cross_entropy',
    'cross_entropy2',
}


#NOTE(zhiqiu): similar as paddle.fluid.contrib.mixed_precision.fp16_lists.AutoMixedPrecisionLists._update_list
# The reason why not use AutoMixedPrecisionLists is that custom_black_varnames is not suitable for imperative mode.
def _update_list(custom_white_list, custom_black_list):
    """
    Update black and white list according to users' custom list.
    """
    _white_list = copy.copy(white_list)
    _black_list = copy.copy(black_list)
    if custom_white_list and custom_black_list:
        for op_name in custom_white_list:
            if op_name in custom_black_list:
                raise ValueError("Custom white list overlap "
                                 "custom black list")
    if custom_white_list:
        for op_name in custom_white_list:
            if op_name in _black_list:
                _black_list.remove(op_name)
            _white_list.add(op_name)
    if custom_black_list:
        for op_name in custom_black_list:
            if op_name in _white_list:
                _white_list.remove(op_name)
            _black_list.add(op_name)
    return _white_list, _black_list


@signature_safe_contextmanager
@dygraph_only
def autocast(enable=True, custom_white_list=None, custom_black_list=None):

    if enable and not core.is_compiled_with_cuda():
        warnings.warn(
            'Auto Mixed Precision can only be enabled with Paddle compiled with CUDA.'
        )
        #enable = False
    tracer = _dygraph_tracer()
    if not tracer:
        raise Exception(
            "current_tracer is None, maybe it is not in imperative mode.")

    _white_list = white_list
    _black_list = black_list
    if custom_white_list or custom_black_list:
        _white_list, _black_list = _update_list(custom_white_list,
                                                custom_black_list)

    if tracer:
        original_val = tracer._enable_autocast
        original_white_list, original_black_list = tracer._get_amp_op_list()

        tracer._enable_autocast = enable
        tracer._set_amp_op_list(_white_list, _black_list)
        original_white_list, original_black_list = tracer._get_amp_op_list()

    try:
        yield
    finally:
        if tracer:
            tracer._enable_autocast = original_val
            tracer._set_amp_op_list(original_white_list, original_black_list)
