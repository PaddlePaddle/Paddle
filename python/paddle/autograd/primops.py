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
        for k in ('shape', 'axis', 'indexes') if k in helper.kwargs
    }

    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type=optype, inputs={'X': x}, outputs={'Y': out}, attrs=attrs)
    return out


# Each primitive op is given a Python constructor for sake of convenience.
def fill_const(value, shape, dtype, out=None):
    attrs = {'value': value, 'shape': shape, 'dtype': dtype}
    helper = LayerHelper('fill_const_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type=helper.layer_type, outputs={'Out': out}, attrs=attrs)
    return out


def neg(x, out=None):
    return _simple_unop(LayerHelper('neg_p', **locals()))


def add(x, y, out=None):
    return _simple_binop(LayerHelper('add_p', **locals()))


def sub(x, y, out=None):
    return _simple_binop(LayerHelper('sub_p', **locals()))


def mul(x, y, out=None):
    return _simple_binop(LayerHelper('mul_p', **locals()))


def div(x, y, out=None):
    return _simple_binop(LayerHelper('div_p', **locals()))


def sqrt(x, out=None):
    return _simple_unop(LayerHelper('sqrt_p', **locals()))


def tanh(x, out=None):
    return _simple_unop(LayerHelper('tanh_p', **locals()))


def reshape(x, shape, out=None):
    return _manipulation_unop(LayerHelper('reshape_p', **locals()))


def broadcast(x, shape, out=None):
    return _manipulation_unop(LayerHelper('broadcast_p', **locals()))


def transpose(x, axis=None, out=None):
    return _manipulation_unop(LayerHelper('transpose_p', **locals()))


def split(x, num=None, sections=None, axis=0, outs=None):
    assert num is None ^ sections is None
    assert sections is None ^ isinstance(sections, (list, tuple))

    n = len(sections) if num is None else num
    attrs = {'axis': axis}
    if num is not None:
        attrs['num'] = num
    else:
        attrs['sections'] = sections

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


def concat(xs, axis=0, out=None):
    assert isinstance(xs, (list, tuple)) and len(xs) > 0
    attrs = {'axis': axis}
    helper = LayerHelper('split_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=xs[0].dtype)
    helper.append_op(
        type=helper.layer_type,
        inputs={'XS': xs},
        outputs={'Y': out},
        attrs=attrs)
    return out


def reduce(x, axis, keepdim, out=None):
    assert isinstance(axis, (int, tuple, list))
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


def matmul(x, y, out=None):
    return _simple_binop(LayerHelper('matmul_p', **locals()))


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


def slice_assign(x, y, axis, starts, ends, strides, out=None):
    assert len(y.shape) == len(starts) == len(ends) == len(strides)
    assert len(y.shape) <= len(x.shape)

    attrs = {'axis': axis, 'starts': starts, 'ends': ends, 'strides': strides}
    helper = LayerHelper('slice_assign_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type=helper.layer_type,
        inputs={'X': x,
                'Y': y},
        outputs={'Y': out},
        attrs=attrs)
    return out


def gather(x, indextensor, axis, out=None):
    attrs = {'axis': axis}
    helper = LayerHelper('gather_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type=helper.layer_type,
        inputs={'X': x,
                'Index': indextensor},
        outputs={'Y': out},
        attrs=attrs)
    return out


def scatter_add(x, y, indextensor, axis, out=None):
    assert y.shape == indextensor.shape
    attrs = {'axis': axis}
    helper = LayerHelper('scatter_add_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type=helper.layer_type,
        inputs={'X': x,
                'Y': y,
                'Index': indextensor},
        outputs={'Z': out},
        attrs=attrs)
    return out


if __name__ == '__main__':
    paddle.enable_static()
    A = paddle.rand([2, 3])
    B = paddle.rand([3, 2])
