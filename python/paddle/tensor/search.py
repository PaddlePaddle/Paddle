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

# TODO: define searching & indexing functions of a tensor  
# __all__ = ['argmax',
#            'argmin',
#            'argsort',
#            'has_inf',
#            'has_nan',
#            'masked_select',
#            'topk',
#            'where',
#            'index_select',
#            'nonzero',
#            'sort']
from __future__ import print_function

from ..fluid.layer_helper import LayerHelper
from ..fluid.layers import cast, where, slice

__all__ = ['index_select', 'nonzero']


def index_select(input, index, dim):
    helper = LayerHelper("index_select", **locals())
    out = helper.create_variable_for_type_inference(input.dtype)
    helper.append_op(
        type='index_select',
        inputs={'X': input,
                'Index': index},
        outputs={'Out': out},
        attrs={'dim': dim})
    return out


def nonzero(inputs, as_tuple=False):
    cast_inputs = cast(inputs, 'bool')
    outs = where(cast_inputs)
    if as_tuple:
        list_out = []
        shape = inputs.shape
        rank = len(shape)
        if rank == 1:
            list_out.append(outs)
        else:
            for i in range(rank):
                list_out.append(
                    slice(
                        outs, axes=[rank - 1], starts=[i], ends=[i + 1]))
        return tuple(list_out)
    else:
        return outs
