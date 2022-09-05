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
import os
import paddle

from pathlib import Path
from typing import Optional, Tuple, Union


class AudioInfo:
    """ Audio info, return type of backend info function """

    def __init__(self, sample_rate: int, num_samples: int, num_channels: int,
                 bits_per_sample: int, encoding: str):
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.bits_per_sample = bits_per_sample
        self.encoding = encoding


def load(filepath: Union[str, Path],
         frame_offset: int = 0,
         num_frames: int = -1,
         normalize: bool = True,
         channels_first: bool = True,
         format: Optional[str] = None) -> Tuple[paddle.Tensor, int]:
    """
       load the given audio file.
    """
    raise RuntimeError("No audio I/O backend is available.")


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
    """
      Save the given audio.
    """
    raise RuntimeError("No audio I/O backend is available.")


def info(filepath: str) -> AudioInfo:
    """ 
      Get signal information of an audio file. 
    """
    raise RuntimeError("No audio I/O backend is available.")
