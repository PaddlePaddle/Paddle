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
from paddle.fluid import unique_name, core
from paddle.fluid.framework import default_main_program, default_startup_program
from paddle.fluid.layer_helper import LayerHelper
from .primreg import REGISTER_FN


def _simple_unop(helper):
    optype = helper.layer_type
    x, out = tuple(map(helper.kwargs.get, ('x', 'out')))
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(type=optype, inputs={'X': x}, outputs={'Y': out}, attrs={})
    return out


def _simple_binop(helper):
    optype = helper.layer_type
    x, y, out = tuple(map(helper.kwargs.get, ('x', 'y', 'out')))
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type=optype, inputs={'X': x,
                             'Y': y}, outputs={'Z': out}, attrs={})
    return out


def _manipulation_unop(helper):
    optype = helper.layer_type
    x, out = tuple(map(helper.kwargs.get, ('x', 'out')))

    attrs = {
        k: helper.kwargs[k]
        for k in ('shape', 'axis', 'index') if k in helper.kwargs
    }

    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type=optype, inputs={'X': x}, outputs={'Y': out}, attrs=attrs)
    return out


# Each primitive op is given a Python constructor for sake of convenience.
def fill_const(value, shape, dtype, out=None):
    attrs = {'value': value, 'shape': shape, 'dtype': dtype}
    helper = LayerHelper('fill_constant_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type=helper.layer_type, outputs={'Y': out}, attrs=attrs)
    return out


def neg(x, out=None):
    zero = fill_const(0.0, x.shape, x.dtype)
    return sub(zero, x)


def set_value(x, y, axis, starts, ends, strides, out):
    assert x is out, "x and out should be the same Tensor in set_value"
    attrs = {'axes': axis, 'starts': starts, 'ends': ends, 'steps': strides}
    helper = LayerHelper('set_value', **locals())
    helper.append_op(
        type=helper.layer_type,
        inputs={'Input': x,
                'ValueTensor': y},
        outputs={'Out': out},
        attrs=attrs)
    return out


@REGISTER_FN('add_p', 'X', 'Y', 'Z')
def add(x, y, out=None):
    return _simple_binop(LayerHelper('add_p', **locals()))


@REGISTER_FN('sub_p', 'X', 'Y', 'Z')
def sub(x, y, out=None):
    return _simple_binop(LayerHelper('sub_p', **locals()))


@REGISTER_FN('mul_p', 'X', 'Y', 'Z')
def mul(x, y, out=None):
    return _simple_binop(LayerHelper('mul_p', **locals()))


@REGISTER_FN('div_p', 'X', 'Y', 'Z')
def div(x, y, out=None):
    return _simple_binop(LayerHelper('div_p', **locals()))


@REGISTER_FN('sqrt_p', 'X', 'Y')
def sqrt(x, out=None):
    return _simple_unop(LayerHelper('sqrt_p', **locals()))


@REGISTER_FN('tanh_p', 'X', 'Y')
def tanh(x, out=None):
    return _simple_unop(LayerHelper('tanh_p', **locals()))


@REGISTER_FN('reshape_p', 'X', 'Y')
def reshape(x, shape, out=None):
    return _manipulation_unop(LayerHelper('reshape_p', **locals()))


@REGISTER_FN('broadcast_p', 'X', 'Y')
def broadcast(x, shape, out=None):
    return _manipulation_unop(LayerHelper('broadcast_p', **locals()))


@REGISTER_FN('transpose_p', 'X', 'Y')
def transpose(x, axis=None, out=None):
    return _manipulation_unop(LayerHelper('transpose_p', **locals()))


@REGISTER_FN('split_p', 'X', 'YS')
def split(x, num_or_sections, axis=0, outs=None):
    if isinstance(num_or_sections, (list, tuple)):
        n = len(num_or_sections)
    else:
        assert isinstance(num_or_sections, int)
        n = num_or_sections

    attrs = {'num_or_sections': num_or_sections, 'axis': axis}

    helper = LayerHelper('split_p', **locals())
    if outs is None:
        outs = [
            helper.create_variable_for_type_inference(dtype=x.dtype)
            for i in range(n)
        ]
    helper.append_op(
        type=helper.layer_type,
        inputs={'X': x},
        outputs={'YS': outs},
        attrs=attrs)
    return outs


@REGISTER_FN('concat_p', 'XS', 'Y')
def concat(xs, axis=0, out=None):
    assert isinstance(xs, (list, tuple)) and len(xs) > 0
    attrs = {'axis': axis}
    helper = LayerHelper('concat_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=xs[0].dtype)
    helper.append_op(
        type=helper.layer_type,
        inputs={'XS': xs},
        outputs={'Y': out},
        attrs=attrs)
    return out


@REGISTER_FN('reduce_p', 'X', 'Y')
def reduce(x, axis, keepdim=False, out=None):
    assert isinstance(axis, (tuple, list))
    assert isinstance(keepdim, bool)

    attrs = {'axis': axis, 'keepdim': keepdim}

    helper = LayerHelper('reduce_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type=helper.layer_type,
        inputs={'X': x},
        outputs={'Y': out},
        attrs=attrs)
    return out


@REGISTER_FN('matmul_p', 'X', 'Y', 'Z')
def matmul(x, y, out=None):
    return _simple_binop(LayerHelper('matmul_p', **locals()))


@REGISTER_FN('slice_select_p', 'X', 'Y')
def slice_select(x, axis, starts, ends, strides, out=None):
    assert isinstance(axis, (list, tuple)), (
        f'Argument type error. `axis` is supposed to be int, list or'
        f' tuple but found {type(axis)}.')
    assert isinstance(starts, (list, tuple))
    assert isinstance(ends, (list, tuple))
    assert len(axis) == len(starts) == len(ends) == len(strides)

    attrs = {'axis': axis, 'starts': starts, 'ends': ends, 'strides': strides}
    helper = LayerHelper('slice_select_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type=helper.layer_type,
        inputs={'X': x},
        outputs={'Y': out},
        attrs=attrs)
    return out


@REGISTER_FN('slice_assign_p', 'X', 'Y', 'Z')
def slice_assign(x, y, axis, starts, ends, strides, out=None):
    assert len(starts) == len(ends) == len(strides) == len(axis)
    assert len(y.shape) == len(x.shape)

    attrs = {'axis': axis, 'starts': starts, 'ends': ends, 'strides': strides}
    helper = LayerHelper('slice_assign_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type=helper.layer_type,
        inputs={'X': x,
                'Y': y},
        outputs={'Z': out},
        attrs=attrs)
    return out


@REGISTER_FN('gather_p', 'X', 'Y')
def gather(x, indextensor, axis, out=None):
    attrs = {'axis': axis}
    helper = LayerHelper('gather_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type=helper.layer_type,
        inputs={'X': x,
                'IndexTensor': indextensor},
        outputs={'Y': out},
        attrs=attrs)
    return out


@REGISTER_FN('scatter_add_p', 'X', 'Y', 'IndexTensor', 'Z')
def scatter_add(x, y, indextensor, axis, out=None):
    assert len(x.shape) == len(y.shape)
    assert len(indextensor.shape) == 1
    assert y.shape[axis] == indextensor.shape[0]
    attrs = {'axis': axis}
    helper = LayerHelper('scatter_add_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type=helper.layer_type,
        inputs={'X': x,
                'Y': y,
                'IndexTensor': indextensor},
        outputs={'Z': out},
        attrs=attrs)
    return out
