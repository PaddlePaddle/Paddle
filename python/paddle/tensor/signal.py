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

from typing import Optional

import paddle

from .attribute import is_complex, is_floating_point
from .fft import fft_r2c, fft_c2r, fft_c2c
from ..fluid.data_feeder import check_variable_and_dtype
from ..fluid.framework import in_dygraph_mode
from ..fluid.layer_helper import LayerHelper
from .. import _C_ops

__all__ = [
    'frame',
    'overlap_add',
    'stft',
    'istft',
]


def frame(x, frame_length, hop_length, axis=-1, name=None):
    '''
        TODO(chenxiaojie06): Doc of frame.
    '''
    if axis not in [0, -1]:
        raise ValueError(f'Unexpected axis: {axis}. It should be 0 or -1.')

    if not isinstance(frame_length, int) or frame_length < 0:
        raise ValueError(
            f'Unexpected frame_length: {frame_length}. It should be an positive integer.'
        )

    if not isinstance(hop_length, int) or hop_length < 0:
        raise ValueError(
            f'Unexpected hop_length: {hop_length}. It should be an positive integer.'
        )

    if frame_length > x.shape[axis]:
        raise ValueError(
            f'Attribute frame_length should be less equal than sequence length, '
            f'but got ({frame_length}) > ({x.shape[axis]}).')

    op_type = 'frame'

    if in_dygraph_mode():
        attrs = ('frame_length', frame_length, 'hop_length', hop_length, 'axis',
                 axis)
        op = getattr(_C_ops, op_type)
        out = op(x, *attrs)
    else:
        check_variable_and_dtype(
            x, 'x', ['int32', 'int64', 'float16', 'float32',
                     'float64'], op_type)
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype=dtype)
        helper.append_op(
            type=op_type,
            inputs={'X': x},
            attrs={
                'frame_length': frame_length,
                'hop_length': hop_length,
                'axis': axis
            },
            outputs={'Out': out})
    return out


def overlap_add(x, hop_length, axis=-1, name=None):
    '''
        TODO(chenxiaojie06): Doc of overlap_add.
    '''
    if axis not in [0, -1]:
        raise ValueError(f'Unexpected axis: {axis}. It should be 0 or -1.')

    if not isinstance(hop_length, int) or hop_length < 0:
        raise ValueError(
            f'Unexpected hop_length: {hop_length}. It should be an positive integer.'
        )

    op_type = 'overlap_add'

    if in_dygraph_mode():
        attrs = ('hop_length', hop_length, 'axis', axis)
        op = getattr(_C_ops, op_type)
        out = op(x, *attrs)
    else:
        check_variable_and_dtype(
            x, 'x', ['int32', 'int64', 'float16', 'float32',
                     'float64'], op_type)
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype=dtype)
        helper.append_op(
            type=op_type,
            inputs={'X': x},
            attrs={'hop_length': hop_length,
                   'axis': axis},
            outputs={'Out': out})
    return out


def stft(x,
         n_fft,
         hop_length=None,
         win_length=None,
         window=None,
         center=True,
         pad_mode='reflect',
         normalized=False,
         onesided=True,
         name=None):
    '''
        TODO(chenxiaojie06): Doc of stft.
    '''
    check_variable_and_dtype(
        x, 'x', ['float16', 'float32', 'float64', 'complex64', 'complex128'],
        'stft')

    x_rank = len(x.shape)
    assert x_rank in [1, 2], \
        f'x should be a 1D or 2D real tensor, but got rank of x is {x_rank}'

    if x_rank == 1:  # (batch, seq_length)
        x = x.unsqueeze(0)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    assert hop_length > 0, \
        f'hop_length should be > 0, but got {hop_length}.'

    if win_length is None:
        win_length = n_fft

    assert 0 < n_fft <= x.shape[-1], \
        f'n_fft should be in (0, seq_length({x.shape[-1]})], but got {n_fft}.'

    assert 0 < win_length <= n_fft, \
        f'win_length should be in (0, n_fft({n_fft})], but got {win_length}.'

    if window is not None:
        assert len(window.shape) == 1 and len(window) == win_length, \
            f'expected a 1D window tensor of size equal to win_length({win_length}), but got window with shape {window.shape}.'
    else:
        window = paddle.ones(shape=(win_length, ), dtype=x.dtype)

    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        window = paddle.nn.functional.pad(window,
                                          pad=[pad_left, pad_right],
                                          mode='constant')

    if center:
        assert pad_mode in ['constant', 'reflect'], \
            'pad_mode should be "reflect" or "constant", but got "{}".'.format(pad_mode)

        pad_length = n_fft // 2
        # FIXME: pad does not supprt complex input.
        x = paddle.nn.functional.pad(x.unsqueeze(-1),
                                     pad=[pad_length, pad_length],
                                     mode=pad_mode,
                                     data_format="NLC").squeeze(-1)

    x_frames = frame(x=x, frame_length=n_fft, hop_length=hop_length, axis=-1)
    x_frames = x_frames.transpose(
        perm=[0, 2,
              1])  # switch n_fft to last dim, egs: (batch, num_frames, n_fft)
    x_frames = x_frames * window

    norm = 'ortho' if normalized else 'backward'
    if is_complex(x_frames):
        assert not onesided, \
            'onesided should be False when input or window is a complex Tensor.'

    if not is_complex(x):
        out = fft_r2c(
            x=x_frames,
            n=None,
            axis=-1,
            norm=norm,
            forward=True,
            onesided=onesided,
            name=name)
    else:
        out = fft_c2c(
            x=x_frames, n=None, axis=-1, norm=norm, forward=True, name=name)

    out = out.transpose(perm=[0, 2, 1])  # (batch, n_fft, num_frames)

    if x_rank == 1:
        out.squeeze_(0)

    return out


def istft(x,
          n_fft,
          hop_length=None,
          win_length=None,
          window=None,
          center=True,
          normalized=False,
          onesided=True,
          length=None,
          return_complex=False,
          name=None):
    '''
        TODO(chenxiaojie06): Doc of istft.
    '''
    check_variable_and_dtype(x, 'x', ['complex64', 'complex128'], 'istft')

    x_rank = len(x.shape)
    assert x_rank in [2, 3], \
        'x should be a 2D or 3D complex tensor, but got rank of x is {}'.format(x_rank)

    if x_rank == 2:  # (batch, n_fft, n_frames)
        x = x.unsqueeze(0)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    if win_length is None:
        win_length = n_fft

    # Assure no gaps between frames.
    assert 0 < hop_length <= win_length, \
        'hop_length should be in (0, win_length({})], but got {}.'.format(win_length, hop_length)

    assert 0 < win_length <= n_fft, \
        'win_length should be in (0, n_fft({})], but got {}.'.format(n_fft, win_length)

    n_frames = x.shape[-1]
    fft_size = x.shape[-2]

    if onesided:
        assert (fft_size == n_fft // 2 + 1), \
            'fft_size should be equal to n_fft // 2 + 1({}) when onesided is True, but got {}.'.format(n_fft // 2 + 1, fft_size)
    else:
        assert (fft_size == n_fft), \
            'fft_size should be equal to n_fft({}) when onesided is False, but got {}.'.format(n_fft, fft_size)

    if window is not None:
        assert len(window.shape) == 1 and len(window) == win_length, \
            'expected a 1D window tensor of size equal to win_length({}), but got window with shape {}.'.format(win_length, window.shape)
    else:
        window = paddle.ones(shape=(win_length, ))

    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        window = paddle.nn.functional.pad(window,
                                          pad=[pad_left, pad_right],
                                          mode='constant')

    x = x.transpose(
        perm=[0, 2,
              1])  # switch n_fft to last dim, egs: (batch, num_frames, n_fft)
    norm = 'ortho' if normalized else 'backward'

    if return_complex:
        assert not onesided, \
            'onesided should be False when input(output of istft) or window is a complex Tensor.'

        out = fft_c2c(x=x, n=None, axis=-1, norm=norm, forward=False, name=None)
    else:
        assert not is_complex(window), \
            'Data type of window should not be complex when return_complex is False.'

        if onesided is False:
            x = x[:, :, :n_fft // 2 + 1]
        out = fft_c2r(x=x, n=None, axis=-1, norm=norm, forward=False, name=None)

    out = overlap_add(
        x=(out * window).transpose(
            perm=[0, 2, 1]),  # (batch, n_fft, num_frames)
        hop_length=hop_length,
        axis=-1)  # (batch, seq_length)

    # FIXME: Use paddle.square when it supports complex tensor.
    window_envelop = overlap_add(
        x=paddle.tile(
            x=window * window, repeat_times=[n_frames, 1]).transpose(
                perm=[1, 0]),  # (n_fft, num_frames)
        hop_length=hop_length,
        axis=-1)  # (seq_length, )

    if length is None:
        if center:
            out = out[:, (n_fft // 2):-(n_fft // 2)]
            window_envelop = window_envelop[(n_fft // 2):-(n_fft // 2)]
    else:
        if center:
            start = n_fft // 2
        else:
            start = 0

        out = out[:, start:start + length]
        window_envelop = window_envelop[start:start + length]

    # Check whether the Nonzero Overlap Add (NOLA) constraint is met.
    if window_envelop.abs().min().item() < 1e-11:
        raise ValueError(
            'Abort istft because Nonzero Overlap Add (NOLA) condition failed. For more information about NOLA constraint please see `scipy.signal.check_NOLA`(https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.check_NOLA.html).'
        )

    out = out / window_envelop

    if x_rank == 2:
        out.squeeze_(0)

    return out
