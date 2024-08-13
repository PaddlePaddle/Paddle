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

from __future__ import annotations

import wave
from typing import TYPE_CHECKING, BinaryIO

import numpy as np

import paddle

from .backend import AudioInfo

if TYPE_CHECKING:
    from pathlib import Path

    from paddle import Tensor


def _error_message():
    package = "paddleaudio"
    warn_msg = (
        "only PCM16 WAV supported. \n"
        "if want support more other audio types, please "
        f"manually installed (usually with `pip install {package}`). \n "
        "and use paddle.audio.backends.set_backend('soundfile') to set audio backend"
    )
    return warn_msg


def info(filepath: str | BinaryIO) -> AudioInfo:
    """Get signal information of input audio file.

    Args:
        filepath: audio path or file object.

    Returns:
        AudioInfo: info of the given audio.

    Example:
        .. code-block:: python

            >>> import os
            >>> import paddle

            >>> sample_rate = 16000
            >>> wav_duration = 0.5
            >>> num_channels = 1
            >>> num_frames = sample_rate * wav_duration
            >>> wav_data = paddle.linspace(-1.0, 1.0, int(num_frames)) * 0.1
            >>> waveform = wav_data.tile([num_channels, 1])
            >>> base_dir = os.getcwd()
            >>> filepath = os.path.join(base_dir, "test.wav")

            >>> paddle.audio.save(filepath, waveform, sample_rate)
            >>> wav_info = paddle.audio.info(filepath)
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
        err_msg = _error_message()
        raise NotImplementedError(err_msg)

    channels = file_.getnchannels()
    sample_rate = file_.getframerate()
    sample_frames = file_.getnframes()  # audio frame
    bits_per_sample = file_.getsampwidth() * 8
    encoding = "PCM_S"  # default WAV encoding, only support
    file_obj.close()
    return AudioInfo(
        sample_rate, sample_frames, channels, bits_per_sample, encoding
    )


def load(
    filepath: str | Path,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
) -> tuple[Tensor, int]:
    """Load audio data from file. load the audio content start form frame_offset, and get num_frames.

    Args:
        frame_offset: from 0 to total frames,
        num_frames: from -1 (means total frames) or number frames which want to read,
        normalize:
            if True: return audio which norm to (-1, 1), dtype=float32
            if False: return audio with raw data, dtype=int16

        channels_first:
            if True: return audio with shape (channels, time)

    Return:
        Tuple[paddle.Tensor, int]: (audio_content, sample rate)

    Examples:
        .. code-block:: python

            >>> import os
            >>> import paddle

            >>> sample_rate = 16000
            >>> wav_duration = 0.5
            >>> num_channels = 1
            >>> num_frames = sample_rate * wav_duration
            >>> wav_data = paddle.linspace(-1.0, 1.0, int(num_frames)) * 0.1
            >>> waveform = wav_data.tile([num_channels, 1])
            >>> base_dir = os.getcwd()
            >>> filepath = os.path.join(base_dir, "test.wav")

            >>> paddle.audio.save(filepath, waveform, sample_rate)
            >>> wav_data_read, sr = paddle.audio.load(filepath)
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
        err_msg = _error_message()
        raise NotImplementedError(err_msg)

    channels = file_.getnchannels()
    sample_rate = file_.getframerate()
    frames = file_.getnframes()  # audio frame

    audio_content = file_.readframes(frames)
    file_obj.close()

    # default_subtype = "PCM_16", only support PCM16 WAV
    audio_as_np16 = np.frombuffer(audio_content, dtype=np.int16)
    audio_as_np32 = audio_as_np16.astype(np.float32)
    if normalize:
        # dtype = "float32"
        audio_norm = audio_as_np32 / (2**15)
    else:
        # dtype = "int16"
        audio_norm = audio_as_np32

    waveform = np.reshape(audio_norm, (frames, channels))
    if num_frames != -1:
        waveform = waveform[frame_offset : frame_offset + num_frames, :]
    waveform = paddle.to_tensor(waveform)
    if channels_first:
        waveform = paddle.transpose(waveform, perm=[1, 0])
    return waveform, sample_rate


def save(
    filepath: str,
    src: Tensor,
    sample_rate: int,
    channels_first: bool = True,
    encoding: str | None = None,
    bits_per_sample: int | None = 16,
) -> None:
    """
    Save audio tensor to file.

    Args:
        filepath: saved path
        src: the audio tensor
        sample_rate: the number of samples of audio per second.
        channels_first: src channel information
            if True, means input tensor is (channels, time)
            if False, means input tensor is (time, channels)
        encoding: audio encoding format, wave_backend only support PCM16 now.
        bits_per_sample: bits per sample, wave_backend only support 16 bits now.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> sample_rate = 16000
            >>> wav_duration = 0.5
            >>> num_channels = 1
            >>> num_frames = sample_rate * wav_duration
            >>> wav_data = paddle.linspace(-1.0, 1.0, int(num_frames)) * 0.1
            >>> waveform = wav_data.tile([num_channels, 1])
            >>> filepath = "./test.wav"

            >>> paddle.audio.save(filepath, waveform, sample_rate)
    """
    assert src.ndim == 2, "Expected 2D tensor"

    audio_numpy = src.numpy()

    # change src shape to (time, channels)
    if channels_first:
        audio_numpy = np.transpose(audio_numpy)

    channels = audio_numpy.shape[1]

    # only support PCM16
    if bits_per_sample not in (None, 16):
        raise ValueError("Invalid bits_per_sample, only support 16 bit")

    sample_width = int(bits_per_sample / 8)  # 2

    if src.dtype == paddle.float32:
        audio_numpy = (audio_numpy * (2**15)).astype("<h")

    with wave.open(filepath, 'w') as f:
        f.setnchannels(channels)
        f.setsampwidth(sample_width)
        f.setframerate(sample_rate)
        f.writeframes(audio_numpy.tobytes())
