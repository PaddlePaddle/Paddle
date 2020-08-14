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

from paddle.fluid.data_feeder import check_variable_and_dtype, check_type
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import Variable

__all__ = ['amp_check_finite_and_scale']


def amp_check_finite_and_scale(x, scale, name=None):
    check_type(x, 'x', (Variable, tuple, list), 'amp_check_finite_and_scale')
    if isinstance(x, list) or isinstance(x, tuple):
        if len(x) > 0:
            for e in x:
                check_variable_and_dtype(e, "x", ['float32', 'float64'],
                                         'amp_check_finite_and_scale')
    else:
        check_variable_and_dtype(x, "x", ['float32', 'float64'],
                                 'amp_check_finite_and_scale')

    helper = LayerHelper("amp_check_finite_and_scale", **locals())
    found_inf = helper.create_variable_for_type_inference(dtype='bool')

    inputs = {'X': x, 'Scale': scale}
    outputs = {'Out': x, 'FoundInfinite': found_inf}
    helper.append_op(
        type='amp_check_finite_and_scale', inputs=inputs, outputs=outputs)

    return found_inf
