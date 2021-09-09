# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from .attribute import is_complex, is_floating_point
from ..fluid.framework import in_dygraph_mode
from .. import _C_ops
from ..fluid.data_feeder import check_variable_and_dtype
from ..fluid.layer_helper import LayerHelper


def _resize_fft_input(x, s, axes):
    if len(s) != len(axes):
        raise ValueError("length of `s` should equals length of `axes`.")
    shape = x.shape
    ndim = x.ndim

    axes_to_pad = []
    paddings = []
    axes_to_slice = []
    slices = []
    for i, axis in axes:
        if shape[axis] < s[i]:
            axes_to_pad.append(axis)
            paddings.append(s[i] - shape[axis])
        elif shape[axis] > s[i]:
            axes_to_slice.append(axis)
            slices.append((0, s[i]))

    if axes_to_slice:
        x = paddle.slice(
            x,
            axes_to_slice,
            starts=[item[0] for item in slices],
            ends=[item[1] for item in slices])
    if axes_to_pad:
        padding_widths = [0] * (2 * ndim)
        for axis, pad in zip(axes_to_pad, paddings):
            padding_widths[2 * axis + 1] = pad
        x = paddle.nn.functional.pad(x, padding_widths)
    return x


# public APIs 1d
def fft(x, n=None, axis=-1, norm="backward", name=None):
    if is_floating_point(x):
        return fft_r2c(x, n, axis, norm, forward=True, onesided=False)
    else:
        return fft_c2c(x, n, axis, norm, forward=True)


def ifft(x, n=None, axis=-1, norm="backward", name=None):
    if is_floating_point(x):
        return fft_r2c(x, n, axis, norm, forward=False, onesided=False)
    else:
        return fft_c2c(x, n, axis, norm, forward=False)


def rfft(x, n=None, axis=-1, norm="backward", name=None):
    return fft_r2c(x, n, axis, norm, forward=True, onesided=True)


def irfft(x, n=None, axis=-1, norm="backward", name=None):
    return fft_c2r(x, n, axis, norm, forward=False)


def hfft(x, n=None, axis=-1, norm="backward", name=None):
    return fft_c2r(x, n, axis, norm, forward=True)


def ihfft(x, n=None, axis=-1, norm="backward", name=None):
    return fft_r2c(x, n, axis, norm, forward=False, onesided=True)


# public APIs nd
def fftn(x, s=None, axes=None, norm="backward", name=None):
    if is_floating_point(x):
        return fftn_r2c(x, s, axes, norm, forward=True, onesided=False)
    else:
        return fftn_c2c(x, s, axes, norm, forward=True)


def ifftn(x, s=None, axes=None, norm="backward", name=None):
    if is_floating_point(x):
        return fftn_r2c(x, s, axes, norm, forward=False, onesided=False)
    else:
        return fftn_c2c(x, s, axes, norm, forward=False)


def rfftn(x, s=None, axes=None, norm="backward", name=None):
    return fftn_r2c(x, s, axes, norm, forward=True, onesided=True)


def irfftn(x, s=None, axes=None, norm="backward", name=None):
    return fftn_c2r(x, s, axes, norm, forward=False)


def hfftn(x, s=None, axes=None, norm="backward", name=None):
    return fftn_c2r(x, s, axes, norm, forward=True)


def ihfftn(x, s=None, axes=None, norm="backward", name=None):
    return fftn_r2c(x, s, axes, norm, forward=False, onesided=True)


## public APIs 2d
def fft2(x, s=None, axes=(-2, -1), norm="backward", name=None):
    return fftn(x, s, axes, norm, name)


def ifft2(x, s=None, axes=(-2, -1), norm="backward", name=None):
    return ifftn(x, s, axes, norm, name)


def rfft2(x, s=None, axes=(-2, -1), norm="backward", name=None):
    return rfftn(x, s, axes, norm, name)


def irfft2(x, s=None, axes=(-2, -1), norm="backward", name=None):
    return irfftn(x, s, axes, norm, name)


def hfft2(x, s=None, axes=(-2, -1), norm="backward", name=None):
    return hfftn(x, s, axes, norm, name)


def ihfft2(x, s=None, axes=(-2, -1), norm="backward", name=None):
    return ihfftn(x, s, axes, norm, name)


# public APIs utilities
def fftfreq(n, d=1.0, dtype=None, name=None):
    dtype = paddle.framework.get_default_dtype()
    val = 1.0 / (n * d)
    pos_max = (n + 1) // 2
    neg_max = n // 2
    indices = paddle.arange(-neg_max, pos_max, dtype=dtype, name=name)
    indices = paddle.roll(indices, -neg_max, name=name)
    return indices * val


def rfftfreq(n, d=1.0, dtype=None, name=None):
    dtype = paddle.framework.get_default_dtype()
    val = 1.0 / (n * d)
    pos_max = 1 + n // 2
    indices = paddle.arange(0, pos_max, dtype=dtype, name=name)
    return indices * val


def fftshift(x, axes=None, name=None):
    shape = paddle.shape(x)
    if axes is None:
        # shift all axes
        rank = paddle.rank(x).reshape([1])
        axes = axes or paddle.arange(0, rank)
        shifts = [size // 2 for size in shape]
    elif isinstance(axes, int):
        shifts = shape[axes] // 2
    else:
        shifts = [shape[ax] // 2 for ax in axes]
    return paddle.roll(x, shifts, axes, name=name)


def ifftshift(x, axes=None, name=None):
    shape = paddle.shape(x)
    if axes is None:
        # shift all axes
        rank = paddle.rank(x).reshape([1])
        axes = axes or paddle.arange(0, rank)
        shifts = [-size // 2 for size in shape]
    elif isinstance(axes, int):
        shifts = -shape[axes] // 2
    else:
        shifts = [-shape[ax] // 2 for ax in axes]
    return paddle.roll(x, shifts, axes, name=name)


# internal functions
def fft_c2c(x, n, axis, norm, forward):
    # TODO, move error checking to operators
    if norm not in ['forward', 'backward', 'ortho']:
        raise ValueError(
            "Unexpected norm: {}. Norm should be forward, backward or ortho".
            format(norm))
    rank = x.ndim
    if axis < -rank or axis >= rank:
        raise ValueError(
            "Invalid axis. Input's ndim is {}, axis should be [-{}, {})".format(
                rank, rank, rank))
    if axis < 0:
        axis += rank
    axes = [axis]
    if n is not None:
        x = _resize_fft_input(x, [n], axes)
    op_type = 'fft_c2c'

    if in_dygraph_mode():
        attrs = ('axes', axes, 'normalization', norm, 'forward', forward)
        out = getattr(_C_ops, op_type)(x, *attrs)
    else:
        inputs = {'X': [x], }
        attrs = {'axes': axes, 'normalization': norm, 'forward': forward}
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 op_type)
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
    return out


def fft_r2c(x, n, axis, norm, forward, onesided):
    if norm not in ['forward', 'backward', 'ortho']:
        raise ValueError(
            "Unexpected norm: {}. Norm should be forward, backward or ortho".
            format(norm))
    rank = x.ndim
    if axis < -rank or axis >= rank:
        raise ValueError(
            "Invalid axis. Input's ndim is {}, axis should be [-{}, {})".format(
                rank, rank, rank))
    if axis < 0:
        axis += rank
    axes = [axis]
    if n is not None:
        x = _resize_fft_input(x, [n], axes)
    op_type = 'fft_r2c'

    if in_dygraph_mode():
        attrs = ('axes', axes, 'normalization', norm, 'forward', forward,
                 'onesided', onesided)
        out = getattr(_C_ops, op_type)(x, *attrs)
    else:
        inputs = {'X': [x], }
        attrs = {
            'axes': axes,
            'normalization': norm,
            'forward': forward,
            'onesided': onesided,
        }
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 op_type)
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
    return out


def fft_c2r(x, n, axis, norm, forward):
    # TODO, move error checking to operators
    if norm not in ['forward', 'backward', 'ortho']:
        raise ValueError(
            "Unexpected norm: {}. Norm should be forward, backward or ortho".
            format(norm))
    rank = x.ndim
    if axis < -rank or axis >= rank:
        raise ValueError(
            "Invalid axis. Input's ndim is {}, axis should be [-{}, {})".format(
                rank, rank, rank))
    if axis < 0:
        axis += rank
    axes = [axis]
    if n is not None:
        x = _resize_fft_input(x, [n // 2 + 1], axes)
    op_type = 'fft_c2r'

    if in_dygraph_mode():
        if n is not None:
            attrs = ('axes', axes, 'normalization', norm, 'forward', forward,
                     'last_dim_size', n)
        else:
            attrs = ('axes', axes, 'normalization', norm, 'forward', forward)
        out = getattr(_C_ops, op_type)(x, *attrs)
    else:
        inputs = {'X': [x], }
        attrs = {'axes': axes, 'normalization': norm, 'forward': forward}
        if n is not None:
            attrs['last_dim_size'] = n
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 op_type)
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
    return out


def fftn_c2c(x, s, axes, norm, forward):
    if norm not in ['forward', 'backward', 'ortho']:
        raise ValueError(
            "Unexpected norm: {}. Norm should be forward, backward or ortho".
            format(norm))
    rank = x.ndim
    if axes is None:
        if s is None:
            axes = list(range(rank))
            s = x.shape
        else:
            fft_ndims = len(s)
            axes = list(range(rank - fft_ndims, rank))
    else:
        axes = list(axes)
        for i in range(len(axes)):
            if axes[i] < -rank or axes[i] >= rank:
                raise ValueError(
                    "Invalid axis. Input's ndim is {}, axis should be [-{}, {})".
                    format(rank, rank, rank))
            if axes[i] < 0:
                axes[i] += rank
        axes = axes
        axes.sort()
        if s is None:
            shape = x.shape
            s = [shape[axis] for axis in axes]
    x = _resize_fft_input(x, s, axes)
    op_type = 'fft_c2c'

    if in_dygraph_mode():
        attrs = ('axes', axes, 'normalization', norm, 'forward', forward)
        out = getattr(_C_ops, op_type)(x, *attrs)
    else:
        inputs = {'X': [x], }
        attrs = {'axes': axes, 'normalization': norm, 'forward': forward}
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 op_type)
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
    return out


def fftn_r2c(x, s, axes, norm, forward, onesided):
    if norm not in ['forward', 'backward', 'ortho']:
        raise ValueError(
            "Unexpected norm: {}. Norm should be forward, backward or ortho".
            format(norm))
    rank = x.ndim
    if axes is None:
        if s is None:
            axes = list(range(rank))
            s = x.shape
        else:
            s = list(s)
            fft_ndims = len(s)
            axes = list(range(rank - fft_ndims, rank))
    else:
        axes = list(axes)
        for i in range(len(axes)):
            if axes[i] < -rank or axes[i] >= rank:
                raise ValueError(
                    "Invalid axis. Input's ndim is {}, axis should be [-{}, {})".
                    format(rank, rank, rank))
            if axes[i] < 0:
                axes[i] += rank
        if s is None:
            shape = x.shape
            s = [shape[axis] for axis in axes]
    x = _resize_fft_input(x, s, axes)
    op_type = 'fft_r2c'

    if in_dygraph_mode():
        attrs = ('axes', axes, 'normalization', norm, 'forward', forward,
                 'onesided', onesided)
        out = getattr(_C_ops, op_type)(x, *attrs)
    else:
        inputs = {'X': [x], }
        attrs = {
            'axes': axes,
            'normalization': norm,
            'forward': forward,
            'onesided': onesided,
        }
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 op_type)
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)

    return out


def fftn_c2r(x, s, axes, norm, forward):
    if norm not in ['forward', 'backward', 'ortho']:
        raise ValueError(
            "Unexpected norm: {}. Norm should be forward, backward or ortho".
            format(norm))
    s = list(s) if s is not None else s
    rank = x.ndim
    if axes is None:
        if s is None:
            axes = list(range(rank))
        else:
            fft_ndims = len(s)
            axes = list(range(rank - fft_ndims, rank))
    else:
        axes = list(axes)
        for i in range(len(axes)):
            if axes[i] < -rank or axes[i] >= rank:
                raise ValueError(
                    "Invalid axis. Input's ndim is {}, axis should be [-{}, {})".
                    format(rank, rank, rank))
            if axes[i] < 0:
                axes[i] += rank
        axes[:-1] = sorted(axes[:-1])
    if s is not None:
        fft_input_shape = [s]
        fft_input_shape[-1] = fft_input_shape[-1] // 2 + 1
        x = _resize_fft_input(x, fft_input_shape, axes)
    op_type = 'fft_c2r'

    if in_dygraph_mode():
        if s:
            attrs = ('axes', axes, 'normalization', norm, 'forward', forward,
                     'last_dim_size', s[-1])
        attrs = ('axes', axes, 'normalization', norm, 'forward', forward)
        out = getattr(_C_ops, op_type)(x, *attrs)
    else:
        inputs = {'X': [x], }
        attrs = {'axes': axes, 'normalization': norm, 'forward': forward}
        if s:
            attrs["last_dim_size"] = s[-1]
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 op_type)
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
    return out
