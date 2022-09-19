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

import os
import paddle

import wave
import warnings
import numpy as np
from pathlib import Path

from typing import Optional, Tuple, Union
from .backend import AudioInfo

__all__ = ['load', 'info', 'save']


def info(filepath: str, format: Optional[str] = None) -> AudioInfo:
    """Get signal information of an audio file.
    only support WAV, PCM16

    Args:
        filepath (path-like object or file-like object):
            Source of audio data.
        format (str or None, optional):
            Not used.

    Returns:
        AudioInfo: info of the given audio.

    """
    if hasattr(filepath, 'read'):
        file_obj = filepath
    else:
        file_obj = open(filepath, 'rb')

    try:
        file_ = wave.open(file_obj)
    except wave.Error:
        file_obj.seek(0)
        file_obj.close()
        raise NotImplementedError(f"Unsupported wave type")

    channels = file_.getnchannels()
    sample_rate = file_.getframerate()
    sample_frames = file_.getnframes()  # audio frame
    bits_per_sample = file_.getsampwidth() * 8
    encoding = "PCM_S"  # default WAV encoding, only support
    file_obj.close()
    return AudioInfo(sample_rate, sample_frames, channels, bits_per_sample,
                     encoding)


def load(filepath: Union[str, Path],
         frame_offset: int = 0,
         num_frames: int = -1,
         normalize: bool = True,
         channels_first: bool = True,
         format: Optional[str] = None) -> Tuple[paddle.Tensor, int]:
    """Load audio data from file.

    Note:
    the wave backend implement by wave.
    Only support wav, 16-bit signed integer

    By default (``normalize=True``, ``channels_first=True``), this function returns Tensor with
    ``float32`` dtype and the shape of `[channel, time]`.
    The samples are normalized to fit in the range of ``[-1.0, 1.0]``.

    ``normalize`` parameter has no effect on 32-bit floating-point WAV.
    For these formats, this function always returns ``float32`` Tensor with values normalized to
    ``[-1.0, 1.0]``.

    Args:
        filepath (path-like object or file-like object):
            Source of audio data.
        frame_offset (int, optional):
            Number of frames to skip before start reading data.
        num_frames (int, optional):
            ``-1`` reads all the remaining samples,
            starting from ``frame_offset``.
        normalize (bool, optional):
            normalize to (-1, 1)
        channels_first (bool, optional):
            When True, the returned Tensor has dimension `[channel, time]`.
            Otherwise, the returned Tensor's dimension is `[time, channel]`.
        format (str or None, optional):
            Not used.

    Returns:
        (paddle.Tensor, int): Resulting Tensor and sample rate.
        If ``channels_first=True``, return `[channel, time]` 
        else return [time, channel]`.
    """
    if hasattr(filepath, 'read'):
        file_obj = filepath
    else:
        file_obj = open(filepath, 'rb')

    try:
        file_ = wave.open(file_obj)
    except wave.Error:
        file_obj.seek(0)
        file_obj.close()
        raise NotImplementedError(f"Unsupported wave type, only PCM16 support")

    channels = file_.getnchannels()
    sample_rate = file_.getframerate()
    frames = file_.getnframes()  # audio frame

    audio_content = file_.readframes(frames)
    file_obj.close()

    #default_subtype = "PCM_16" # only support PCM16 WAV
    audio_as_np16 = np.frombuffer(audio_content, dtype=np.int16)
    audio_as_np32 = audio_as_np16.astype(np.float32)
    if normalize:
        #dtype = "float32"
        audio_norm = audio_as_np32 / (2**15)
    else:
        #dtype = "int16"
        audio_norm = audio_as_np32

    waveform = np.reshape(audio_norm, (frames, channels))
    if num_frames != -1:
        waveform = waveform[frame_offset:frame_offset + num_frames, :]
    waveform = paddle.to_tensor(waveform)
    if channels_first:
        waveform = paddle.transpose(waveform, perm=[1, 0])
    return waveform, sample_rate


def save(
    filepath: str,
    src: paddle.Tensor,
    sample_rate: int,
    channels_first: bool = True,
    compression: Optional[float] = None,
    format: Optional[str] = None,
    encoding: Optional[str] = None,
    bits_per_sample: Optional[int] = None,
):
    """Save audio data to file.

    Note:
        Only support the following formats,

        * WAV

            * 16-bit signed integer

    Note:
        ``filepath`` argument is intentionally annotated as ``str`` only, even though it accepts
        ``pathlib.Path`` object as well.

    Args:
        filepath (str or pathlib.Path): Path to audio file.
        src (paddle.Tensor): Audio data to save. must be 2D tensor.
        sample_rate (int): sampling rate
        channels_first (bool, optional): If ``True``, the given tensor is interpreted as `[channel, time]`,
            otherwise `[time, channel]`.
        compression (float of None, optional): Not used.
        format (str or None, optional): wav only.
        encoding (str or None, optional): default PCM.

        bits_per_sample (int or None, optional): 16bit only

    Supported formats/encodings/bit depth/compression are:
    ``"wav"``
        - 16-bit signed integer PCM
    """
    # the src shape is (time, channels), not channel_first
    if src.ndim != 2:
        raise ValueError(f"Expected 2D Tensor, got {src.ndim}D.")

    if compression is not None:
        warnings.warn("The argument compression is silently ignored.")

    if format not in (None, "WAV", "wav"):
        raise RuntimeError("`format` is only support WAV.")

    audio_numpy = src.numpy()
    if channels_first:
        audio_numpy = np.transpose(audio_numpy)

    channels = audio_numpy.shape[0]

    # only support PCM16
    if bits_per_sample not in (None, 16):
        raise ValueError("Invalid bits_per_sample.")

    sample_width = 2  #bits_per_sample / 8

    if src.dtype == paddle.float32:
        audio_numpy = (audio_numpy * (2**15)).astype("<h")

    with wave.open(filepath, 'w') as f:
        f.setnchannels(channels)
        f.setsampwidth(sample_width)
        f.setframerate(sample_rate)
        f.writeframes(audio_numpy.tobytes())
