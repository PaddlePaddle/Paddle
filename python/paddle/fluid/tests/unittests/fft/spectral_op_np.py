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

import numpy as np
from functools import partial
from numpy import asarray
from numpy.fft._pocketfft import _raw_fft, _raw_fftnd, _get_forward_norm, _get_backward_norm, _cook_nd_args


def _fftc2c(a, n=None, axis=-1, norm=None, forward=None):
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    if forward:
        inv_norm = _get_forward_norm(n, norm)
    else:
        inv_norm = _get_backward_norm(n, norm)
    output = _raw_fft(a, n, axis, False, forward, inv_norm)
    return output


def _fftr2c(a, n=None, axis=-1, norm=None, forward=None):
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    if forward:
        inv_norm = _get_forward_norm(n, norm)
    else:
        inv_norm = _get_backward_norm(n, norm)
    output = _raw_fft(a, n, axis, True, True, inv_norm)
    if not forward:
        output = output.conj()
    return output


def _fftc2r(a, n=None, axis=-1, norm=None, forward=None):
    a = asarray(a)
    if n is None:
        n = (a.shape[axis] - 1) * 2
    if forward:
        inv_norm = _get_forward_norm(n, norm)
    else:
        inv_norm = _get_backward_norm(n, norm)
    output = _raw_fft(a.conj() if forward else a, n, axis, True, False,
                      inv_norm)
    return output


def fft_c2c(x, axes, normalization, forward):
    f = partial(_fftc2c, forward=forward)
    y = _raw_fftnd(x, s=None, axes=axes, function=f, norm=normalization)
    return y


def fft_c2c_backward(dy, axes, normalization, forward):
    f = partial(_fftc2c, forward=forward)
    dx = _raw_fftnd(dy, s=None, axes=axes, function=f, norm=normalization)
    return dx


def fft_r2c(x, axes, normalization, forward, onesided):
    a = asarray(x)
    s, axes = _cook_nd_args(a, axes=axes)
    if onesided:
        a = _fftr2c(a, s[-1], axes[-1], normalization, forward)
        for ii in range(len(axes) - 1):
            a = _fftc2c(a, s[ii], axes[ii], normalization, forward)
    else:
        a = fft_c2c(x, axes, normalization, forward)
    return a


def fft_r2c_backward(dy, x, axes, normalization, forward, onesided):
    a = dy
    if not onesided:
        a = fft_c2c_backward(a, axes, normalization, forward).real
    else:
        pad_widths = [(0, 0)] * a.ndim
        last_axis = axes[-1]
        if last_axis < 0:
            last_axis += a.ndim
        last_dim_size = a.shape[last_axis]
        pad_widths[last_axis] = (0, x.shape[last_axis] - last_dim_size)
        a = np.pad(a, pad_width=pad_widths)
        a = fft_c2c_backward(a, axes, normalization, forward).real
    return a


def fft_c2r(x, axes, normalization, forward, last_dim_size):
    a = asarray(x)
    s, axes = _cook_nd_args(a, axes=axes, invreal=1)
    if last_dim_size is not None:
        s[-1] = last_dim_size
    for ii in range(len(axes) - 1):
        a = _fftc2c(a, s[ii], axes[ii], normalization, forward)
    a = _fftc2r(a, s[-1], axes[-1], normalization, forward)
    return a
