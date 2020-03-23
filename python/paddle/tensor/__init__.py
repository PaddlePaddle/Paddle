# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from ..fluid.layer_helper import LayerHelper
from ..fluid import core
from ..fluid.layers import cast, where, slice

__all__ = ["index_select", "nonzero"]


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
        for i in range(rank):
            list_out.append(slice(outs, axes=rank - 1, starts=i, ends=i + 1))
        return tuple(list_out)
    else:
        return outs


def cross(input, other, dim=-1):
    helper = LayerHelper("cross", **locals())
    out = helper.create_variable_for_type_inference(input.dtype)
    helper.append_op(
        type='cross',
        inputs={'X': input,
                'Y': other},
        outputs={'Out': out},
        attrs={'dim': dim})
    return out


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
