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
# limitations under the License

from pathlib import Path
from typing import Optional, Tuple, Union

import paddle


class AudioInfo:
    """Audio info, return type of backend info function"""

    def __init__(
        self,
        sample_rate: int,
        num_samples: int,
        num_channels: int,
        bits_per_sample: int,
        encoding: str,
    ):
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.bits_per_sample = bits_per_sample
        self.encoding = encoding


def info(filepath: str) -> AudioInfo:
    """Get signal information of input audio file.

    Args:
       filepath: audio path or file object.

    Returns:
        AudioInfo: info of the given audio.

    Example:
        .. code-block:: python

            import os
            import paddle

            sample_rate = 16000
            wav_duration = 0.5
            num_channels = 1
            num_frames = sample_rate * wav_duration
            wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1
            waveform = wav_data.tile([num_channels, 1])
            base_dir = os.getcwd()
            filepath = os.path.join(base_dir, "test.wav")

            paddle.audio.save(filepath, waveform, sample_rate)
            wav_info = paddle.audio.info(filepath)
    """
    # for API doc
    raise NotImplementedError("please set audio backend")


def load(
    filepath: Union[str, Path],
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
) -> Tuple[paddle.Tensor, int]:
    """Load audio data from file.Load the audio content start form frame_offset, and get num_frames.

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

            import os
            import paddle

            sample_rate = 16000
            wav_duration = 0.5
            num_channels = 1
            num_frames = sample_rate * wav_duration
            wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1
            waveform = wav_data.tile([num_channels, 1])
            base_dir = os.getcwd()
            filepath = os.path.join(base_dir, "test.wav")

            paddle.audio.save(filepath, waveform, sample_rate)
            wav_data_read, sr = paddle.audio.load(filepath)
    """
    # for API doc
    raise NotImplementedError("please set audio backend")


def save(
    filepath: str,
    src: paddle.Tensor,
    sample_rate: int,
    channels_first: bool = True,
    encoding: Optional[str] = None,
    bits_per_sample: Optional[int] = 16,
):
    """
    Save audio tensor to file.

    Args:
        filepath: saved path
        src: the audio tensor
        sample_rate: the number of samples of audio per second.
        channels_first: src channel information
            if True, means input tensor is (channels, time)
            if False, means input tensor is (time, channels)
        encoding:encoding format, wave_backend only support PCM16 now.
        bits_per_sample: bits per sample, wave_backend only support 16 bits now.

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle

            sample_rate = 16000
            wav_duration = 0.5
            num_channels = 1
            num_frames = sample_rate * wav_duration
            wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1
            waveform = wav_data.tile([num_channels, 1])
            filepath = "./test.wav"

            paddle.audio.save(filepath, waveform, sample_rate)
    """
    # for API doc
    raise NotImplementedError("please set audio backend")
