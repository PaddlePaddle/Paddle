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

import re
import sys
import unittest

import numpy as np
from numpy import fft
from numpy.lib.stride_tricks import as_strided
import paddle
import scipy.signal

paddle.set_default_dtype('float64')

DEVICES = [paddle.CPUPlace()]
if paddle.is_compiled_with_cuda():
    DEVICES.append(paddle.CUDAPlace(0))
TEST_CASE_NAME = 'test_case'

# Constrain STFT block sizes to 256 KB
MAX_MEM_BLOCK = 2**8 * 2**10


def fix_length(data, size, axis=-1, **kwargs):
    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]

    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)

    return data


def tiny(x):
    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
            x.dtype, np.complexfloating):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny


def normalize(S, norm=np.inf, axis=0, threshold=None, fill=None):
    # Avoid div-by-zero
    if threshold is None:
        threshold = tiny(S)

    elif threshold <= 0:
        raise Exception("threshold={} must be strictly "
                        "positive".format(threshold))

    if fill not in [None, False, True]:
        raise Exception("fill={} must be None or boolean".format(fill))

    if not np.all(np.isfinite(S)):
        raise Exception("Input must be finite")

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(np.float64)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise Exception("Cannot normalize with norm=0 and fill=True")

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag**norm, axis=axis, keepdims=True)**(1.0 / norm)

        if axis is None:
            fill_norm = mag.size**(-1.0 / norm)
        else:
            fill_norm = mag.shape[axis]**(-1.0 / norm)

    elif norm is None:
        return S

    else:
        raise Exception("Unsupported norm: {}".format(repr(norm)))

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm


def __window_ss_fill(x, win_sq, n_frames, hop_length):  # pragma: no cover
    """Helper function for window sum-square calculation."""

    n = len(x)
    n_fft = len(win_sq)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample +
                     n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]


def window_sumsquare(
    window,
    n_frames,
    hop_length=512,
    win_length=None,
    n_fft=2048,
    dtype=np.float32,
    norm=None,
):
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length)
    win_sq = normalize(win_sq, norm=norm)**2
    win_sq = pad_center(win_sq, n_fft)

    # Fill the envelope
    __window_ss_fill(x, win_sq, n_frames, hop_length)

    return x


def dtype_c2r(d, default=np.float32):
    mapping = {
        np.dtype(np.complex64): np.float32,
        np.dtype(np.complex128): np.float64,
    }

    # If we're given a real type already, return it
    dt = np.dtype(d)
    if dt.kind == "f":
        return dt

    # Otherwise, try to map the dtype.
    # If no match is found, return the default.
    return np.dtype(mapping.get(np.dtype(d), default))


def dtype_r2c(d, default=np.complex64):
    mapping = {
        np.dtype(np.float32): np.complex64,
        np.dtype(np.float64): np.complex128,
    }

    # If we're given a complex type already, return it
    dt = np.dtype(d)
    if dt.kind == "c":
        return dt

    # Otherwise, try to map the dtype.
    # If no match is found, return the default.
    return np.dtype(mapping.get(dt, default))


def frame(x, frame_length, hop_length, axis=-1):
    if not isinstance(x, np.ndarray):
        raise Exception("Input must be of type numpy.ndarray, "
                        "given type(x)={}".format(type(x)))

    if x.shape[axis] < frame_length:
        raise Exception("Input is too short (n={:d})"
                        " for frame_length={:d}".format(x.shape[axis],
                                                        frame_length))

    if hop_length < 1:
        raise Exception("Invalid hop_length: {:d}".format(hop_length))

    if axis == -1 and not x.flags["F_CONTIGUOUS"]:
        print("librosa.util.frame called with axis={} "
              "on a non-contiguous input. This will result in a copy.".format(
                  axis))
        x = np.asfortranarray(x)
    elif axis == 0 and not x.flags["C_CONTIGUOUS"]:
        print("librosa.util.frame called with axis={} "
              "on a non-contiguous input. This will result in a copy.".format(
                  axis))
        x = np.ascontiguousarray(x)

    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    strides = np.asarray(x.strides)

    new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize

    if axis == -1:
        shape = list(x.shape)[:-1] + [frame_length, n_frames]
        strides = list(strides) + [hop_length * new_stride]

    elif axis == 0:
        shape = [n_frames, frame_length] + list(x.shape)[1:]
        strides = [hop_length * new_stride] + list(strides)

    else:
        raise Exception("Frame axis={} must be either 0 or -1".format(axis))

    return as_strided(x, shape=shape, strides=strides)


def pad_center(data, size, axis=-1, **kwargs):
    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise Exception(("Target size ({:d}) must be "
                         "at least input size ({:d})").format(size, n))

    return np.pad(data, lengths, **kwargs)


def get_window(window, Nx, fftbins=True):
    if callable(window):
        return window(Nx)

    elif isinstance(window, (str, tuple)) or np.isscalar(window):
        # TODO: if we add custom window functions in librosa, call them here

        return scipy.signal.get_window(window, Nx, fftbins=fftbins)

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        raise Exception("Window size mismatch: "
                        "{:d} != {:d}".format(len(window), Nx))
    else:
        raise Exception("Invalid window specification: {}".format(window))


def __overlap_add(y, ytmp, hop_length):
    # numba-accelerated overlap add for inverse stft
    # y is the pre-allocated output buffer
    # ytmp is the windowed inverse-stft frames
    # hop_length is the hop-length of the STFT analysis

    n_fft = ytmp.shape[0]
    for frame in range(ytmp.shape[1]):
        sample = frame * hop_length
        y[sample:(sample + n_fft)] += ytmp[:, frame]


def stft(x,
         n_fft=2048,
         hop_length=None,
         win_length=None,
         window="hann",
         center=True,
         pad_mode="reflect"):
    y = x
    input_rank = len(y.shape)
    if input_rank == 2:
        assert y.shape[0] == 1  # Only 1d input supported in librosa
        y = y.squeeze(0)
    dtype = None

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Pad the time series so that frames are centered
    if center:
        if n_fft > y.shape[-1]:
            print("n_fft={} is too small for input signal of length={}".format(
                n_fft, y.shape[-1]))

        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    elif n_fft > y.shape[-1]:
        raise Exception(
            "n_fft={} is too large for input signal of length={}".format(
                n_fft, y.shape[-1]))

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    if dtype is None:
        dtype = dtype_r2c(y.dtype)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order="F")

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:,
                    bl_s:bl_t] = fft.rfft(fft_window * y_frames[:, bl_s:bl_t],
                                          axis=0)

    if input_rank == 2:
        stft_matrix = np.expand_dims(stft_matrix, 0)

    return stft_matrix


def istft(
    x,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    length=None,
):

    stft_matrix = x
    input_rank = len(stft_matrix.shape)
    if input_rank == 3:
        assert stft_matrix.shape[0] == 1  # Only 2d input supported in librosa
        stft_matrix = stft_matrix.squeeze(0)
    dtype = None

    n_fft = 2 * (stft_matrix.shape[0] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    ifft_window = get_window(window, win_length, fftbins=True)

    # Pad out to match n_fft, and add a broadcasting axis
    ifft_window = pad_center(ifft_window, n_fft)[:, np.newaxis]

    # For efficiency, trim STFT frames according to signal length if available
    if length:
        if center:
            padded_length = length + int(n_fft)
        else:
            padded_length = length
        n_frames = min(stft_matrix.shape[1],
                       int(np.ceil(padded_length / hop_length)))
    else:
        n_frames = stft_matrix.shape[1]

    expected_signal_len = n_fft + hop_length * (n_frames - 1)

    if dtype is None:
        dtype = dtype_c2r(stft_matrix.dtype)

    y = np.zeros(expected_signal_len, dtype=dtype)

    n_columns = MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize)
    n_columns = min(n_columns, 1)

    frame = 0
    for bl_s in range(0, n_frames, n_columns):
        bl_t = min(bl_s + n_columns, n_frames)

        # invert the block and apply the window function
        ytmp = ifft_window * fft.irfft(stft_matrix[:, bl_s:bl_t], axis=0)

        # Overlap-add the istft block starting at the i'th frame
        __overlap_add(y[frame * hop_length:], ytmp, hop_length)

        frame += bl_t - bl_s

    # Normalize by sum of squared window
    ifft_window_sum = window_sumsquare(
        window,
        n_frames,
        win_length=win_length,
        n_fft=n_fft,
        hop_length=hop_length,
        dtype=dtype,
    )

    approx_nonzero_indices = ifft_window_sum > tiny(ifft_window_sum)
    y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    if length is None:
        # If we don't need to control length, just do the usual center trimming
        # to eliminate padded data
        if center:
            y = y[int(n_fft // 2):-int(n_fft // 2)]
    else:
        if center:
            # If we're centering, crop off the first n_fft//2 samples
            # and then trim/pad to the target length.
            # We don't trim the end here, so that if the signal is zero-padded
            # to a longer duration, the decay is smooth by windowing
            start = int(n_fft // 2)
        else:
            # If we're not centering, start at 0 and trim/pad as necessary
            start = 0

        y = fix_length(y[start:], length)

    if input_rank == 3:
        y = np.expand_dims(y, 0)

    return y


def frame_for_api_test(x, frame_length, hop_length, axis=-1):
    if axis == -1 and not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    elif axis == 0 and not x.flags["F_CONTIGUOUS"]:
        x = np.asfortranarray(x)

    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    strides = np.asarray(x.strides)

    if axis == -1:
        shape = list(x.shape)[:-1] + [frame_length, n_frames]
        strides = list(strides) + [hop_length * x.itemsize]

    elif axis == 0:
        shape = [n_frames, frame_length] + list(x.shape)[1:]
        strides = [hop_length * x.itemsize] + list(strides)

    else:
        raise ValueError("Frame axis={} must be either 0 or -1".format(axis))

    return as_strided(x, shape=shape, strides=strides)


def overlap_add_for_api_test(x, hop_length, axis=-1):
    assert axis in [0, -1], 'axis should be 0/-1.'
    assert len(x.shape) >= 2, 'Input dims shoulb be >= 2.'

    squeeze_output = False
    if len(x.shape) == 2:
        squeeze_output = True
        dim = 0 if axis == -1 else -1
        x = np.expand_dims(x, dim)  # batch

    n_frames = x.shape[axis]
    frame_length = x.shape[1] if axis == 0 else x.shape[-2]

    # Assure no gaps between frames.
    assert 0 < hop_length <= frame_length, \
        f'hop_length should be in (0, frame_length({frame_length})], but got {hop_length}.'

    seq_length = (n_frames - 1) * hop_length + frame_length

    reshape_output = False
    if len(x.shape) > 3:
        reshape_output = True
        if axis == 0:
            target_shape = [seq_length] + list(x.shape[2:])
            x = x.reshape(n_frames, frame_length, np.product(x.shape[2:]))
        else:
            target_shape = list(x.shape[:-2]) + [seq_length]
            x = x.reshape(np.product(x.shape[:-2]), frame_length, n_frames)

    if axis == 0:
        x = x.transpose((2, 1, 0))

    y = np.zeros(shape=[np.product(x.shape[:-2]), seq_length], dtype=x.dtype)
    for i in range(x.shape[0]):
        for frame in range(x.shape[-1]):
            sample = frame * hop_length
            y[i, sample:sample + frame_length] += x[i, :, frame]

    if axis == 0:
        y = y.transpose((1, 0))

    if reshape_output:
        y = y.reshape(target_shape)

    if squeeze_output:
        y = y.squeeze(-1) if axis == 0 else y.squeeze(0)

    return y


def place(devices, key='place'):

    def decorate(cls):
        module = sys.modules[cls.__module__].__dict__
        raw_classes = {
            k: v
            for k, v in module.items() if k.startswith(cls.__name__)
        }

        for raw_name, raw_cls in raw_classes.items():
            for d in devices:
                test_cls = dict(raw_cls.__dict__)
                test_cls.update({key: d})
                new_name = raw_name + '.' + d.__class__.__name__
                module[new_name] = type(new_name, (raw_cls, ), test_cls)
            del module[raw_name]
        return cls

    return decorate


def setUpModule():
    global rtol
    global atol
    # All test case will use float64 for compare percision, refs:
    # https://github.com/PaddlePaddle/Paddle/wiki/Upgrade-OP-Precision-to-Float64
    rtol = {
        'float32': 1e-06,
        'float64': 1e-7,
        'complex64': 1e-06,
        'complex128': 1e-7,
    }
    atol = {
        'float32': 0.0,
        'float64': 0.0,
        'complex64': 0.0,
        'complex128': 0.0,
    }


def tearDownModule():
    pass


def rand_x(dims=1,
           dtype='float64',
           min_dim_len=1,
           max_dim_len=10,
           shape=None,
           complex=False):

    if shape is None:
        shape = [
            np.random.randint(min_dim_len, max_dim_len) for i in range(dims)
        ]
    if complex:
        return np.random.randn(
            *shape).astype(dtype) + 1.j * np.random.randn(*shape).astype(dtype)
    else:
        return np.random.randn(*shape).astype(dtype)


def parameterize(attrs, input_values=None):

    if isinstance(attrs, str):
        attrs = [attrs]
    input_dicts = (attrs if input_values is None else
                   [dict(zip(attrs, vals)) for vals in input_values])

    def decorator(base_class):
        test_class_module = sys.modules[base_class.__module__].__dict__
        for idx, input_dict in enumerate(input_dicts):
            test_class_dict = dict(base_class.__dict__)
            test_class_dict.update(input_dict)

            name = class_name(base_class, idx, input_dict)

            test_class_module[name] = type(name, (base_class, ),
                                           test_class_dict)

        for method_name in list(base_class.__dict__):
            if method_name.startswith("test"):
                delattr(base_class, method_name)
        return base_class

    return decorator


def class_name(cls, num, params_dict):
    suffix = to_safe_name(
        next((v for v in params_dict.values() if isinstance(v, str)), ""))
    if TEST_CASE_NAME in params_dict:
        suffix = to_safe_name(params_dict["test_case"])
    return "{}_{}{}".format(cls.__name__, num, suffix and "_" + suffix)


def to_safe_name(s):
    return str(re.sub("[^a-zA-Z0-9_]+", "_", s))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'frame_length', 'hop_length', 'axis'),
    [
        ('test_1d_input1', rand_x(1, np.float64, shape=[150]), 50, 15, 0),
        ('test_1d_input2', rand_x(1, np.float64, shape=[150]), 50, 15, -1),
        ('test_2d_input1', rand_x(2, np.float64, shape=[150, 8]), 50, 15, 0),
        ('test_2d_input2', rand_x(2, np.float64, shape=[8, 150]), 50, 15, -1),
        ('test_3d_input1', rand_x(3, np.float64, shape=[150, 4, 2]), 50, 15, 0),
        ('test_3d_input2', rand_x(3, np.float64, shape=[4, 2, 150]), 50, 15, -1),
    ]) # yapf: disable
class TestFrame(unittest.TestCase):

    def test_frame(self):
        np.testing.assert_allclose(frame_for_api_test(self.x, self.frame_length,
                                                      self.hop_length,
                                                      self.axis),
                                   paddle.signal.frame(paddle.to_tensor(self.x),
                                                       self.frame_length,
                                                       self.hop_length,
                                                       self.axis),
                                   rtol=rtol.get(str(self.x.dtype)),
                                   atol=atol.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'frame_length', 'hop_length', 'axis'),
    [
        ('test_1d_input1', rand_x(1, np.float64, shape=[150]), 50, 15, 0),
        ('test_1d_input2', rand_x(1, np.float64, shape=[150]), 50, 15, -1),
        ('test_2d_input1', rand_x(2, np.float64, shape=[150, 8]), 50, 15, 0),
        ('test_2d_input2', rand_x(2, np.float64, shape=[8, 150]), 50, 15, -1),
        ('test_3d_input1', rand_x(3, np.float64, shape=[150, 4, 2]), 50, 15, 0),
        ('test_3d_input2', rand_x(3, np.float64, shape=[4, 2, 150]), 50, 15, -1),
    ]) # yapf: disable
class TestFrameStatic(unittest.TestCase):

    def test_frame_static(self):
        paddle.enable_static()
        mp, sp = paddle.static.Program(), paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            input = paddle.static.data('input',
                                       self.x.shape,
                                       dtype=self.x.dtype)
            output = paddle.signal.frame(input, self.frame_length,
                                         self.hop_length, self.axis),
        exe = paddle.static.Executor(self.place)
        exe.run(sp)
        [output] = exe.run(mp, feed={'input': self.x}, fetch_list=[output])
        paddle.disable_static()

        np.testing.assert_allclose(frame_for_api_test(self.x, self.frame_length,
                                                      self.hop_length,
                                                      self.axis),
                                   output,
                                   rtol=rtol.get(str(self.x.dtype)),
                                   atol=atol.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'frame_length', 'hop_length', 'axis', 'expect_exception'),
    [
        ('test_axis', rand_x(1, np.float64, shape=[150]), 50, 15, 2, ValueError),
        ('test_hop_length', rand_x(1, np.float64, shape=[150]), 50, 0, -1, ValueError),
        ('test_frame_length1', rand_x(2, np.float64, shape=[150, 8]), 0, 15, 0, ValueError),
        ('test_frame_length2', rand_x(2, np.float64, shape=[150, 8]), 151, 15, 0, ValueError),
    ]) # yapf: disable
class TestFrameException(unittest.TestCase):

    def test_frame(self):
        with self.assertRaises(self.expect_exception):
            paddle.signal.frame(paddle.to_tensor(self.x), self.frame_length,
                                self.hop_length, self.axis)


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'hop_length', 'axis'),
    [
        ('test_2d_input1', rand_x(2, np.float64, shape=[3, 50]), 4, 0),
        ('test_2d_input2', rand_x(2, np.float64, shape=[50, 3]), 4, -1),
        ('test_3d_input1', rand_x(3, np.float64, shape=[5, 40, 2]), 10, 0),
        ('test_3d_input2', rand_x(3, np.float64, shape=[2, 40, 5]), 10, -1),
        ('test_4d_input1', rand_x(4, np.float64, shape=[8, 12, 5, 3]), 5, 0),
        ('test_4d_input2', rand_x(4, np.float64, shape=[3, 5, 12, 8]), 5, -1),
    ]) # yapf: disable
class TestOverlapAdd(unittest.TestCase):

    def test_overlap_add(self):
        np.testing.assert_allclose(
            overlap_add_for_api_test(self.x, self.hop_length, self.axis),
            paddle.signal.overlap_add(paddle.to_tensor(self.x), self.hop_length,
                                      self.axis),
            rtol=rtol.get(str(self.x.dtype)),
            atol=atol.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'hop_length', 'axis'),
    [
        ('test_2d_input1', rand_x(2, np.float64, shape=[3, 50]), 4, 0),
        ('test_2d_input2', rand_x(2, np.float64, shape=[50, 3]), 4, -1),
        ('test_3d_input1', rand_x(3, np.float64, shape=[5, 40, 2]), 10, 0),
        ('test_3d_input2', rand_x(3, np.float64, shape=[2, 40, 5]), 10, -1),
        ('test_4d_input1', rand_x(4, np.float64, shape=[8, 12, 5, 3]), 5, 0),
        ('test_4d_input2', rand_x(4, np.float64, shape=[3, 5, 12, 8]), 5, -1),
    ]) # yapf: disable
class TestOverlapAddStatic(unittest.TestCase):

    def test_overlap_add_static(self):
        paddle.enable_static()
        mp, sp = paddle.static.Program(), paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            input = paddle.static.data('input',
                                       self.x.shape,
                                       dtype=self.x.dtype)
            output = paddle.signal.overlap_add(input, self.hop_length,
                                               self.axis),
        exe = paddle.static.Executor(self.place)
        exe.run(sp)
        [output] = exe.run(mp, feed={'input': self.x}, fetch_list=[output])
        paddle.disable_static()

        np.testing.assert_allclose(overlap_add_for_api_test(
            self.x, self.hop_length, self.axis),
                                   output,
                                   rtol=rtol.get(str(self.x.dtype)),
                                   atol=atol.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'hop_length', 'axis', 'expect_exception'),
    [
        ('test_axis', rand_x(2, np.float64, shape=[3, 50]), 4, 2, ValueError),
        ('test_hop_length', rand_x(2, np.float64, shape=[50, 3]), -1, -1, ValueError),
    ]) # yapf: disable
class TestOverlapAddException(unittest.TestCase):

    def test_overlap_add(self):
        with self.assertRaises(self.expect_exception):
            paddle.signal.overlap_add(paddle.to_tensor(self.x), self.hop_length,
                                      self.axis)


# ================= STFT
# common args
#   x
#   n_fft,
#   hop_length=None,
#   win_length=None,
#   window=None,
#   center=True,
#   pad_mode='reflect',

# paddle only
#   normalized=False,
#   onesided=True,

# ================= ISTFT
# common args
#    x,
#    hop_length=None,
#    win_length=None,
#    window=None,
#    center=True,
#    length=None,

# paddle only
#    n_fft,
#    normalized=False,
#    onesided=True,
#    return_complex=False,


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n_fft', 'hop_length', 'win_length', 'window', 'center', 'pad_mode', 'normalized', 'onesided'),
    [
        ('test_1d_input', rand_x(1, np.float64, shape=[160000]),
        512, None, None, get_window('hann', 512), True, 'reflect', False, True),
        ('test_2d_input', rand_x(2, np.float64, shape=[1, 160000]),
        512, None, None, get_window('hann', 512), True, 'reflect', False, True),
        ('test_hop_length', rand_x(2, np.float64, shape=[1, 160000]),
        512, 255, None, get_window('hann', 512), True, 'reflect', False, True),
        ('test_win_length', rand_x(2, np.float64, shape=[1, 160000]),
        512, 255, 499, get_window('hann', 499), True, 'reflect', False, True),
        ('test_window', rand_x(2, np.float64, shape=[1, 160000]),
        512, None, None, None, True, 'reflect', False, True),
        ('test_center', rand_x(2, np.float64, shape=[1, 160000]),
        512, None, None, None, False, 'reflect', False, True),
    ])# yapf: disable
class TestStft(unittest.TestCase):

    def test_stft(self):
        if self.window is None:
            win_p = None
            win_l = 'boxcar'  # rectangular window
        else:
            win_p = paddle.to_tensor(self.window)
            win_l = self.window

        np.testing.assert_allclose(
            stft(self.x, self.n_fft, self.hop_length, self.win_length, win_l,
                 self.center, self.pad_mode),
            paddle.signal.stft(paddle.to_tensor(self.x), self.n_fft,
                               self.hop_length, self.win_length, win_p,
                               self.center, self.pad_mode, self.normalized,
                               self.onesided),
            rtol=rtol.get(str(self.x.dtype)),
            atol=atol.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n_fft', 'hop_length', 'win_length', 'window', 'center', 'pad_mode', 'normalized', 'onesided', 'expect_exception'),
    [
        ('test_dims', rand_x(1, np.float64, shape=[1, 2, 3]),
        512, None, None, None, True, 'reflect', False, True, AssertionError),
        ('test_hop_length', rand_x(1, np.float64, shape=[16000]),
        512, 0, None, None, True, 'reflect', False, True, AssertionError),
        ('test_nfft1', rand_x(1, np.float64, shape=[16000]),
        0, None, None, None, True, 'reflect', False, True, AssertionError),
        ('test_nfft2', rand_x(1, np.float64, shape=[16000]),
        16001, None, None, None, True, 'reflect', False, True, AssertionError),
        ('test_win_length', rand_x(1, np.float64, shape=[16000]),
        512, None, 0, None, True, 'reflect', False, True, AssertionError),
        ('test_win_length', rand_x(1, np.float64, shape=[16000]),
        512, None, 513, None, True, 'reflect', False, True, AssertionError),
        ('test_pad_mode', rand_x(1, np.float64, shape=[16000]),
        512, None, None, None, True, 'nonsense', False, True, AssertionError),
        ('test_complex_onesided', rand_x(1, np.float64, shape=[16000], complex=True),
        512, None, None, None, False, 'reflect', False, True, AssertionError),
    ]) # yapf: disable
class TestStftException(unittest.TestCase):

    def test_stft(self):
        if self.window is None:
            win_p = None
        else:
            win_p = paddle.to_tensor(self.window)

        with self.assertRaises(self.expect_exception):
            paddle.signal.stft(paddle.to_tensor(self.x), self.n_fft,
                               self.hop_length, self.win_length, win_p,
                               self.center, self.pad_mode, self.normalized,
                               self.onesided),


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n_fft', 'hop_length', 'win_length', 'window', 'center', 'normalized', 'onesided', 'length', 'return_complex'),
    [
        ('test_2d_input', rand_x(2, np.float64, shape=[257, 471], complex=True),
        512, None, None, get_window('hann', 512), True, False, True, None, False),
        ('test_3d_input', rand_x(3, np.float64, shape=[1, 257, 471], complex=True),
        512, None, None, get_window('hann', 512), True, False, True, None, False),
        ('test_hop_length', rand_x(3, np.float64, shape=[1, 257, 471], complex=True),
        512, 99, None, get_window('hann', 512), True, False, True, None, False),
        ('test_win_length', rand_x(3, np.float64, shape=[1, 257, 471], complex=True),
        512, 99, 299, get_window('hann', 299), True, False, True, None, False),
        ('test_window', rand_x(3, np.float64, shape=[1, 257, 471], complex=True),
        512, None, None, None, True, False, True, None, False),
        ('test_center', rand_x(3, np.float64, shape=[1, 257, 471], complex=True),
        512, None, None, None, False, False, True, None, False),
        ('test_length', rand_x(3, np.float64, shape=[1, 257, 471], complex=True),
        512, None, None, None, False, False, True, 1888, False),
    ]) # yapf: disable
class TestIstft(unittest.TestCase):

    def test_istft(self):
        if self.window is None:
            win_p = None
            win_l = 'boxcar'  # rectangular window
        else:
            win_p = paddle.to_tensor(self.window)
            win_l = self.window

        np.testing.assert_allclose(
            istft(self.x, self.hop_length, self.win_length, win_l, self.center,
                  self.length),
            paddle.signal.istft(paddle.to_tensor(self.x), self.n_fft,
                                self.hop_length, self.win_length, win_p,
                                self.center, self.normalized, self.onesided,
                                self.length, self.return_complex),
            rtol=rtol.get(str(self.x.dtype)),
            atol=atol.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n_fft', 'hop_length', 'win_length', 'window', 'center', 'normalized', 'onesided', 'length', 'return_complex', 'expect_exception'),
    [
        ('test_dims', rand_x(4, np.float64, shape=[1, 2, 3, 4], complex=True),
        512, None, None, get_window('hann', 512), True, False, True, None, False, AssertionError),
        ('test_n_fft', rand_x(3, np.float64, shape=[1, 257, 471], complex=True),
        257, None, None, get_window('hann', 512), True, False, True, None, False, AssertionError),
        ('test_hop_length1', rand_x(3, np.float64, shape=[1, 257, 471], complex=True),
        512, 0, None, get_window('hann', 512), True, False, True, None, False, AssertionError),
        ('test_hop_length2', rand_x(3, np.float64, shape=[1, 257, 471], complex=True),
        512, 513, None, get_window('hann', 512), True, False, True, None, False, AssertionError),
        ('test_win_length1', rand_x(3, np.float64, shape=[1, 257, 471], complex=True),
        512, None, 0, get_window('hann', 512), True, False, True, None, False, AssertionError),
        ('test_win_length2', rand_x(3, np.float64, shape=[1, 257, 471], complex=True),
        512, None, 513, get_window('hann', 512), True, False, True, None, False, AssertionError),
        ('test_onesided1', rand_x(3, np.float64, shape=[1, 257, 471], complex=True),
        20, None, None, get_window('hann', 512), True, False, True, None, False, AssertionError),
        ('test_onesided2', rand_x(3, np.float64, shape=[1, 257, 471], complex=True),
        256, None, None, None, True, False, False, None, False, AssertionError),
        ('test_window', rand_x(3, np.float64, shape=[1, 512, 471], complex=True),
        512, None, 511, get_window('hann', 512), True, False, False, None, False, AssertionError),
        ('test_return_complex1', rand_x(3, np.float64, shape=[1, 257, 471], complex=True),
        512, None, None, get_window('hann', 512), True, False, True, None, True, AssertionError),
        ('test_return_complex2', rand_x(3, np.float64, shape=[1, 257, 471], complex=True),
        512, None, None, rand_x(1, np.float64, shape=[512], complex=True), True, False, True, None, False, AssertionError),
        ('test_NOLA', rand_x(3, np.float64, shape=[1, 257, 471], complex=True),
        512, 512, None, get_window('hann', 512), True, False, True, None, False, ValueError),
    ]) # yapf: disable
class TestIstftException(unittest.TestCase):

    def test_istft(self):
        if self.window is None:
            win_p = None
        else:
            win_p = paddle.to_tensor(self.window)

        with self.assertRaises(self.expect_exception):
            paddle.signal.istft(paddle.to_tensor(self.x), self.n_fft,
                                self.hop_length, self.win_length, win_p,
                                self.center, self.normalized, self.onesided,
                                self.length, self.return_complex),


if __name__ == '__main__':
    unittest.main()
