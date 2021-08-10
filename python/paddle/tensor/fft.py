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


# public APIs 1d
def fft(x, n=None, axis=-1, norm=None, name=None):
    if is_floating_point(x):
        return fft_r2c(x, n, axis, norm, forward=True, onesided=False)
    else:
        return fft_c2c(x, n, axis, norm, forward=True)


def ifft(x, n=None, axis=-1, norm=None, name=None):
    if is_floating_point(x):
        return fft_r2c(x, n, axis, norm, forward=False, onesided=False)
    else:
        return fft_c2c(x, n, axis, norm, forward=False)


def rfft(x, n=None, axis=-1, norm=None, name=None):
    return fft_r2c(x, n, axis, norm, forward=True, onesided=True)


def irfft(x, n=None, axis=-1, norm=None, name=None):
    return fft_c2r(x, n, axis, norm, forward=False)


def hfft(x, n=None, axis=-1, norm=None, name=None):
    return fft_c2r(x, n, axis, norm, forward=True)


def ihfft(x, n=None, axis=-1, norm=None, name=None):
    return fft_r2c(x, n, axis, norm, forward=False, onesided=True)


# public APIs nd
def fftn(x, s=None, axes=None, norm=None, name=None):
    if is_floating_point(x):
        return fft_r2c(x, s, axes, norm, forward=True, onesided=False)
    else:
        return fft_c2c(x, s, axes, norm, forward=True)


def ifftn(x, s=None, axes=None, norm=None, name=None):
    if is_floating_point(x):
        return fft_r2c(x, s, axes, norm, forward=False, onesided=False)
    else:
        return fft_c2c(x, s, axes, norm, forward=False)


def rfftn(x, s=None, axes=None, norm=None, name=None):
    return fft_r2c(x, s, axes, norm, forward=True, onesided=True)


def irfftn(x, s=None, axes=None, norm=None, name=None):
    return fft_c2r(x, s, axes, norm, forward=False)


def hfftn(x, s=None, axes=None, norm=None, name=None):
    return fft_c2r(x, s, axes, norm, forward=True)


def ihfftn(x, s=None, axes=None, norm=None, name=None):
    return fft_r2c(x, s, axes, norm, forward=False, onesided=True)


## public APIs 2d
def fft2(x, s=None, axes=(-2, -1), norm=None, name=None):
    return fftn(x, s, axes, norm, name)


def ifft2(x, s=None, axes=(-2, -1), norm=None, name=None):
    return ifftn(x, s, axes, norm, name)


def rfft2(x, s=None, axes=(-2, -1), norm=None, name=None):
    return rfftn(x, s, axes, norm, name)


def irfft2(x, s=None, axes=(-2, -1), norm=None, name=None):
    return irfftn(x, s, axes, norm, name)


def hfft2(x, s=None, axes=(-2, -1), norm=None, name=None):
    return hfftn(x, s, axes, norm, name)


def ihfft2(x, s=None, axes=(-2, -1), norm=None, name=None):
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
    pass


def fft_r2c(x, n, axis, norm, forward, onesided):
    pass


def fft_c2r(x, n, axis, norm, forward):
    pass


def fftn_c2c(x, s, axes, norm, forward):
    pass


def fftn_r2c(x, s, axes, norm, forward, onesided):
    pass


def fftn_c2r(x, s, axes, norm, forward):
    pass
