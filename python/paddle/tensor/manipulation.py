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

# TODO: define functions to manipulate a tensor  
# __all__ = ['cast',
#            'concat',
#            'expand',
#            'expand_as',
#            'flatten',
#            'gather',
#            'gather_nd',
#            'reshape',
#            'reverse',
#            'scatter',
#            'scatter_nd_add',
#            'scatter_nd',
#            'shard_index',
#            'slice',
#            'split',
#            'squeeze',
#            'stack',
#            'strided_slice',
#            'transpose',
#            'unique',
#            'unique_with_counts',
#            'unsqueeze',
#            'unstack',
#            'flip',
#            'unbind',
#            'roll']
from __future__ import print_function

from ..fluid.layer_helper import LayerHelper

__all__ = ['roll']


def roll(input, shifts, dims=None):
    helper = LayerHelper("roll", **locals())
    out = helper.create_variable_for_type_inference(input.dtype)

    if type(shifts) == int:
        shifts = [shifts]
    if type(dims) == int:
        dims = [dims]
    if dims is None:
        dims = [0]

    helper.append_op(
        type='roll',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={'dims': dims,
               'shifts': shifts})
    return out
