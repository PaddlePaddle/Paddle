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

import functools
import operator

import paddle
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

    helper.append_op(type=optype,
                     inputs={
                         'X': x,
                         'Y': y
                     },
                     outputs={'Z': out},
                     attrs={})
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

    helper.append_op(type=optype,
                     inputs={'X': x},
                     outputs={'Y': out},
                     attrs=attrs)
    return out


# Each primitive op is given a Python constructor for sake of convenience.
def fill_const(value, shape, dtype, out=None):
    attrs = {'value': value, 'shape': shape, 'dtype': dtype}
    helper = LayerHelper('fill_constant_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type=helper.layer_type, outputs={'Y': out}, attrs=attrs)
    return out


def bernoulli(shape, dtype, p, out=None):
    attrs = {'shape': shape, 'dtype': dtype, 'p': p}
    helper = LayerHelper('bernoulli_p', **locals())
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
    helper.append_op(type=helper.layer_type,
                     inputs={
                         'Input': x,
                         'ValueTensor': y
                     },
                     outputs={'Out': out},
                     attrs=attrs)
    return out


def mean(x, axis=None, keepdim=False):
    axes = axis or tuple(range(0, len(x.shape)))
    sum = reduce_sum(x, axis=axes, keepdim=keepdim)
    norm = fill_const(shape=sum.shape,
                      value=functools.reduce(operator.mul,
                                             [x.shape[axis] for axis in axes]),
                      dtype=sum.dtype)
    return div(sum, norm)


def ones(shape, dtype):
    return fill_const(1, shape, dtype)


def zeros(shape, dtype):
    return fill_const(0, shape, dtype)


def batch_norm(x,
               axis,
               gamma,
               beta,
               run_mean,
               run_var,
               eps=1e-5,
               momentum=0.9,
               use_run_stat=False,
               reserve_space=None):
    """batch normalizer.

    Args:
        x (Tensor): A tensor to be normalized.
        axis (int): The features axis.
        gamma (Tensor): The scale factor.
        beta (float): The shift factor.
        run_mean (Tensor): Running mean.
        run_var (Tensor): Running variance.
        eps (float, optional): A value added to the denominator for numerical
            stability. Defaults to 1e-5.
        momentum (float, optional): The value used for the running_mean and
            running_var computation. Can be set to None for cumulative moving
            average (i.e. simple average). Defaults to 0.9.
        use_run_stat (bool, optional): Whether or not using runing statistics.
            Defaults to False.
    """
    reduce_axes = tuple(i for i in range(len(x.shape)) if i != axis)
    stats_shape = tuple(1 if i in reduce_axes else s
                        for i, s in enumerate(x.shape))

    batch_mean = zeros(run_mean.shape, run_mean.dtype)
    batch_var = zeros(run_var.shape, run_var.dtype)

    if not use_run_stat:
        batch_mean = mean(x, reduce_axes, keepdim=True)
        batch_var = mean(square(sub(x, broadcast(batch_mean, x.shape))),
                         reduce_axes,
                         keepdim=True)
        x_hat = div(
            sub(x, broadcast(batch_mean, x.shape)),
            sqrt(
                add(broadcast(batch_var, x.shape),
                    fill_const(eps, x.shape, batch_var.dtype))))

        momentum = fill_const(momentum, run_mean.shape, run_mean.dtype)
        run_mean = add(
            mul(momentum, run_mean),
            mul(sub(ones(run_mean.shape, run_mean.dtype), momentum),
                reshape(batch_mean, run_mean.shape)))
        run_var = add(
            mul(momentum, run_var),
            mul(sub(ones(run_var.shape, run_var.dtype), momentum),
                reshape(batch_var, run_var.shape)))
    else:
        x_hat = div(
            sub(x, broadcast(reshape(run_mean, stats_shape), x.shape)),
            sqrt(
                add(broadcast(reshape(run_var, stats_shape), x.shape),
                    fill_const(eps, x.shape, x.dtype))))
    y = add(mul(broadcast(reshape(gamma, stats_shape), x_hat.shape), x_hat),
            broadcast(reshape(beta, stats_shape), x_hat.shape))

    if reserve_space:
        return run_mean, reserve_space, batch_mean, batch_var, run_var, y
    else:
        return run_mean, batch_mean, batch_var, run_var, y


def square(x):
    return pow(x, fill_const(2., x.shape, x.dtype))


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


@REGISTER_FN('sin_p', 'X', 'Y')
def sin(x, out=None):
    return _simple_unop(LayerHelper('sin_p', **locals()))


@REGISTER_FN('cos_p', 'X', 'Y')
def cos(x, out=None):
    return _simple_unop(LayerHelper('cos_p', **locals()))


@REGISTER_FN('exp_p', 'X', 'Y')
def exp(x, out=None):
    return _simple_unop(LayerHelper('exp_p', **locals()))


@REGISTER_FN('abs_p', 'X', 'Y')
def abs(x, out=None):
    return _simple_unop(LayerHelper('abs_p', **locals()))


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
        if not isinstance(num_or_sections, int):
            raise TypeError(
                f'num_or_sections must be int, but got {type(num_or_sections)}.'
            )
        n = num_or_sections

    attrs = {'num_or_sections': num_or_sections, 'axis': axis}

    helper = LayerHelper('split_p', **locals())
    if outs is None:
        outs = [
            helper.create_variable_for_type_inference(dtype=x.dtype)
            for i in range(n)
        ]
    helper.append_op(type=helper.layer_type,
                     inputs={'X': x},
                     outputs={'YS': outs},
                     attrs=attrs)
    return outs


@REGISTER_FN('concat_p', 'XS', 'Y')
def concat(xs, axis=0, out=None):
    if isinstance(xs, paddle.fluid.framework.Variable):
        xs = [xs]
    attrs = {'axis': axis}
    helper = LayerHelper('concat_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=xs[0].dtype)
    helper.append_op(type=helper.layer_type,
                     inputs={'XS': xs},
                     outputs={'Y': out},
                     attrs=attrs)
    return out


@REGISTER_FN('reduce_sum_p', 'X', 'Y')
def reduce_sum(x, axis=None, keepdim=False, out=None):
    axes = axis or tuple(range(0, len(x.shape)))
    axes = (axes, ) if isinstance(axes, int) else axes
    if not isinstance(axis, (tuple, list)):
        raise TypeError(f'axis must be tuple or list, but got {type(axis)}')
    if not isinstance(keepdim, bool):
        raise TypeError(f'keepdim must be bool, but got {type(keepdim)}')

    attrs = {'axis': axis, 'keepdim': keepdim}
    helper = LayerHelper('reduce_sum_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(type=helper.layer_type,
                     inputs={'X': x},
                     outputs={'Y': out},
                     attrs=attrs)
    return out


@REGISTER_FN('matmul_p', 'X', 'Y', 'Z')
def matmul(x, y, out=None):
    return _simple_binop(LayerHelper('matmul_p', **locals()))


@REGISTER_FN('slice_select_p', 'X', 'Y')
def slice_select(x, axis, starts, ends, strides, out=None):
    if not isinstance(axis, (list, tuple)):
        raise TypeError(f'Argument type error. `axis` is supposed to be list or'
                        f' tuple but found {type(axis)}.')
    if not isinstance(starts, (list, tuple)):
        raise TypeError(
            f'Argument type error. `starts` is supposed to be list or'
            f' tuple but found {type(starts)}.')
    if not isinstance(ends, (list, tuple)):
        raise TypeError(f'Argument type error. `ends` is supposed to be list or'
                        f' tuple but found {type(ends)}.')
    assert len(axis) == len(starts) == len(ends) == len(strides), (
        f'len(axis), len(starts), len(ends) and len(strides) should be equal, '
        f'but len(axis)={len(axis)}, len(starts)={len(starts)}, '
        f'len(ends)={len(ends)} and len(strides)={len(strides)}')

    attrs = {'axis': axis, 'starts': starts, 'ends': ends, 'strides': strides}
    helper = LayerHelper('slice_select_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type=helper.layer_type,
                     inputs={'X': x},
                     outputs={'Y': out},
                     attrs=attrs)
    return out


@REGISTER_FN('slice_assign_p', 'X', 'Y', 'Z')
def slice_assign(x, y, axis, starts, ends, strides, out=None):
    assert len(starts) == len(ends) == len(strides) == len(axis), (
        f'len(starts), len(ends), len(strides) and len(axis) should be equal, '
        f'but len(starts)={len(starts)}, len(ends)={len(ends)}, '
        f'len(strides)={len(strides)} and len(axis)={len(axis)}')
    assert len(y.shape) == len(x.shape), (
        f'len(y.shape) should be equal to len(x.shape), '
        f'but len(y.shape)={len(y.shape)} and len(x.shape)={len(x.shape)}.')

    attrs = {'axis': axis, 'starts': starts, 'ends': ends, 'strides': strides}
    helper = LayerHelper('slice_assign_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type=helper.layer_type,
                     inputs={
                         'X': x,
                         'Y': y
                     },
                     outputs={'Z': out},
                     attrs=attrs)
    return out


@REGISTER_FN('gather_p', 'X', 'IndexTensor', 'Y')
def gather(x, indextensor, axis, out=None):
    attrs = {'axis': axis}
    helper = LayerHelper('gather_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type=helper.layer_type,
                     inputs={
                         'X': x,
                         'IndexTensor': indextensor
                     },
                     outputs={'Y': out},
                     attrs=attrs)
    return out


@REGISTER_FN('scatter_add_p', 'X', 'Y', 'IndexTensor', 'Z')
def scatter_add(x, y, indextensor, axis, out=None):
    assert len(x.shape) == len(y.shape), (
        f'len(x.shape) should be equal to len(y.shape), '
        f'but len(x.shape)={len(x.shape)} and len(y.shape)={len(y.shape)}.')
    assert len(
        indextensor.shape
    ) == 1, f'len(indextensor.shape) must be equal to 1, but got {len(indextensor.shape)}.'
    assert y.shape[axis] == indextensor.shape[0], (
        f'y.shape[axis] should be equal to indextensor.shape[0], '
        f'but y.shape[axis]={y.shape[axis]} and '
        f'indextensor.shape[0]={indextensor.shape[0]}.')
    attrs = {'axis': axis}
    helper = LayerHelper('scatter_add_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type=helper.layer_type,
                     inputs={
                         'X': x,
                         'Y': y,
                         'IndexTensor': indextensor
                     },
                     outputs={'Z': out},
                     attrs=attrs)
    return out


@REGISTER_FN('log_p', 'X', 'Y')
def log(x, out=None):
    return _simple_unop(LayerHelper('log_p', **locals()))


@REGISTER_FN('select_p', 'Condition', 'X', 'Y', 'Z')
def select(cond, x, y, out=None):
    if len(cond.shape) != len(x.shape):
        raise ValueError(
            "len(cond.shape) should be equal to len(x.shape), but len(cond.shape)={} and len(x.shape)={}."
            .format(len(cond.shape), len(x.shape)))

    if len(x.shape) != len(y.shape):
        raise ValueError(
            "len(x.shape) should be equal to len(y.shape), but len(x.shape)={} and len(y.shape)={}."
            .format(len(x.shape), len(y.shape)))

    helper = LayerHelper('select_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type=helper.layer_type,
                     inputs={
                         'Condition': cond,
                         'X': x,
                         'Y': y
                     },
                     outputs={'Z': out})
    return out


@REGISTER_FN('eq_p', 'X', 'Y', 'Z')
def eq(x, y, out=None):
    return _simple_binop(LayerHelper('eq_p', **locals()))


@REGISTER_FN('gt_p', 'X', 'Y', 'Z')
def gt(x, y, out=None):
    return _simple_binop(LayerHelper('gt_p', **locals()))


@REGISTER_FN('ge_p', 'X', 'Y', 'Z')
def ge(x, y, out=None):
    return _simple_binop(LayerHelper('ge_p', **locals()))


@REGISTER_FN('ne_p', 'X', 'Y', 'Z')
def ne(x, y, out=None):
    return _simple_binop(LayerHelper('ne_p', **locals()))


@REGISTER_FN('pow_p', 'X', 'Y', 'Z')
def pow(x, y, out=None):
    return _simple_binop(LayerHelper('pow_p', **locals()))


@REGISTER_FN('max_p', 'X', 'Y', 'Z')
def max(x, y, out=None):
    return _simple_binop(LayerHelper('max_p', **locals()))


@REGISTER_FN('erf_p', 'X', 'Y')
def erf(x, out=None):
    return _simple_unop(LayerHelper('erf_p', **locals()))


@REGISTER_FN('cast_p', 'X', 'Y')
def cast(x, dtype, out=None):
    helper = LayerHelper('cast_p', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type=helper.layer_type,
                     inputs={'X': x},
                     outputs={'Y': out},
                     attrs={'dtype': dtype})
    return out


@REGISTER_FN('rsqrt_p', 'X', 'Y')
def rsqrt(x, out=None):
    return _simple_unop(LayerHelper('rsqrt_p', **locals()))
