# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.fluid.layer_helper import LayerHelper

def _simple_unop(helper):
    optype = helper.layer_type
    x, out = tuple(map(helper.kwargs.get, ('x', 'out')))
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type=optype,
        inputs={'X': x},
        outputs={'Out': out},
        attrs={})
    return out


def _simple_binop(helper):
    optype = helper.layer_type
    x, y, out = tuple(map(helper.kwargs.get, ('x', 'y', 'out')))
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type=optype,
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs={})
    return out


def _manipulation_unop(helper):
    optype = helper.layer_type
    x, out = tuple(map(helper.kwargs.get, ('x', 'out')))
    
    attrs = {k: helper.kwargs[k] for k in ('shape', 'axis', 'indexes') if k in helper.kwargs}

    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    
    attrs = {}

    helper.append_op(
        type=optype,
        inputs={'X': x},
        outputs={'Out': out},
        attrs=attrs)
    return out


# Each primitive op is given a Python constructor for sake of convenience.
def add(x, y, out=None):
    return _simple_binop(LayerHelper('prim_add', **locals()))    


def sub(x, y, out=None):
    return _simple_binop(LayerHelper('prim_sub', **locals()))


def mul(x, y, out=None):
    return _simple_binop(LayerHelper('prim_mul', **locals()))


def div(x, y, out=None):
    return _simple_binop(LayerHelper('prim_div', **locals()))


def sqrt(x, out=None):
    return _simple_unop(LayerHelper('prim_sqrt', **locals()))


def tanh(x, out=None):
    return _simple_unop(LayerHelper('prim_tanh', **locals()))


def reshape(x, shape, out=None):
    return _manipulation_unop(LayerHelper('prim_reshape', **locals()))


def broadcast(x, shape, out=None):
    return _manipulation_unop(LayerHelper('prim_broadcast', **locals()))


def transpose(x, axis=None, out=None):
    return _manipulation_unop(LayerHelper('prim_transpose', **locals()))


def split(x, num_or_sections, axis=0, out=None):
    


def concat(xs, out=None):



def reduce(x, axis, out=None):



