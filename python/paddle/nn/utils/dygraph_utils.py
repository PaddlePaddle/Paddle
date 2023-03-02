# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.framework import dygraph_only


@dygraph_only
def _append_bias_in_dygraph(input, bias=None, axis=1, use_mkldnn=False):
    """Append bias operation in dygraph mode.

        Args:
            input: the input variable.
            bias:  the bias to be appended
            axis:  the axis to perform operation
            use_mkldnn: whether to use mkldnn

    Return the Variable after bias operation
    """
    if bias is None:
        return input

    return _legacy_C_ops.elementwise_add(
        input, bias, 'axis', axis, 'use_mkldnn', use_mkldnn
    )
