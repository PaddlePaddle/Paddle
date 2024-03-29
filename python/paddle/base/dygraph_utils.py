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

from paddle import _legacy_C_ops

from .framework import dygraph_only


@dygraph_only
def _append_activation_in_dygraph(input, act=None, use_cudnn=None):
    """Append activation in dygraph mode.

        Args:
            input: the input variable.
            act: activation type
            use_cudnn: if use cudnn

    Return the Variable after append activation
    """
    if act is None:
        return input

    attrs = ()
    if use_cudnn:
        attrs = ('use_cudnn', use_cudnn)

    act_op = getattr(_legacy_C_ops, act)
    return act_op(input, *attrs)
