# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from . import core
from .framework import dygraph_only


@dygraph_only
def _append_activation_in_dygraph(input,
                                  act=None,
                                  use_cudnn=False,
                                  use_mkldnn=False):
    """Append activation in dygraph mode.

        Args:
            input: the input variable. 
            act: activation type
            use_mkldnn: if use mkldnn
            use_cudnn: if use cudnn

    Return the Variable after append activation
    """
    if not act:
        return input

    attrs = {'use_cudnn': use_cudnn, 'use_mkldnn': use_mkldnn}
    inputs = {"X": [input]}
    act_op = getattr(core.ops, act)
    res = act_op(inputs, attrs)
    return res['Out'][0]


@dygraph_only
def _append_bias_in_dygraph(
        input,
        bias=None,
        axis=1, ):
    """Append bias operation in dygraph mode.

        Args:
            input: the input variable. 
            bias:  the bias to be appended
            axis:  the axis to perform operation

    Return the Variable after bias operation
    """
    if not bias:
        return input

    attrs = {'axis': axis}
    inputs = {'X': [input], 'Y': [bias]}
    outs = core.ops.elementwise_add(inputs, attrs)
    return outs['Out'][0]
