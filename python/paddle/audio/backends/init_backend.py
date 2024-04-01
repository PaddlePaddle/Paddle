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

import sys
import warnings
from typing import List

import paddle

from . import backend, wave_backend


def _check_version(version: str) -> bool:
    # require paddleaudio >= 1.0.2
    ver_arr = version.split('.')
    v0 = int(ver_arr[0])
    v1 = int(ver_arr[1])
    v2 = int(ver_arr[2])
    if v0 < 1:
        return False
    if v0 == 1 and v1 == 0 and v2 <= 1:
        return False
    return True


def list_available_backends() -> List[str]:
    """List available backends, the backends in paddleaudio and the default backend.

    Returns:
        List[str]: The list of available backends.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> sample_rate = 16000
            >>> wav_duration = 0.5
            >>> num_channels = 1
            >>> num_frames = sample_rate * wav_duration
            >>> wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1
            >>> waveform = wav_data.tile([num_channels, 1])
            >>> wav_path = "./test.wav"

            >>> current_backend = paddle.audio.backends.get_current_backend()
            >>> print(current_backend)
            wave_backend

            >>> backends = paddle.audio.backends.list_available_backends()
            >>> # default backends is ['wave_backend']
            >>> # backends is ['wave_backend', 'soundfile'], if have installed paddleaudio >= 1.0.2
            >>> if 'soundfile' in backends:
            ...     paddle.audio.backends.set_backend('soundfile')
            ...
            >>> paddle.audio.save(wav_path, waveform, sample_rate)

    """
    backends = []
    try:
        import paddleaudio
    except ImportError:
        package = "paddleaudio"
        warn_msg = (
            f"Failed importing {package}. \n"
            "only wave_backend(only can deal with PCM16 WAV) supported.\n"
            "if want soundfile_backend(more audio type supported),\n"
            f"please manually installed (usually with `pip install {package} >= 1.0.2`). "
        )
        warnings.warn(warn_msg)

    if "paddleaudio" in sys.modules:
        version = paddleaudio.__version__
        if not _check_version(version):
            err_msg = (
                f"the version of paddleaudio installed is {version},\n"
                "please ensure the paddleaudio >= 1.0.2."
            )
            raise ImportError(err_msg)
        backends = paddleaudio.backends.list_audio_backends()
    backends.append("wave_backend")
    return backends


def get_current_backend() -> str:
    """Get the name of the current audio backend

    Returns:
        str: The name of the current backend,
        the wave_backend or backend imported from paddleaudio

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> sample_rate = 16000
            >>> wav_duration = 0.5
            >>> num_channels = 1
            >>> num_frames = sample_rate * wav_duration
            >>> wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1
            >>> waveform = wav_data.tile([num_channels, 1])
            >>> wav_path = "./test.wav"

            >>> current_backend = paddle.audio.backends.get_current_backend()
            >>> print(current_backend)
            wave_backend

            >>> backends = paddle.audio.backends.list_available_backends()
            >>> # default backends is ['wave_backend']
            >>> # backends is ['wave_backend', 'soundfile'], if have installed paddleaudio >= 1.0.2

            >>> if 'soundfile' in backends:
            ...     paddle.audio.backends.set_backend('soundfile')
            ...
            >>> paddle.audio.save(wav_path, waveform, sample_rate)

    """
    current_backend = None
    if "paddleaudio" in sys.modules:
        import paddleaudio

        current_backend = paddleaudio.backends.get_audio_backend()
        if paddle.audio.load == paddleaudio.load:
            return current_backend
    return "wave_backend"


def set_backend(backend_name: str):
    """Set the backend by one of the list_audio_backend return.

    Args:
        backend (str): one of the list_audio_backend. "wave_backend" is the default. "soundfile" imported from paddleaudio.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> sample_rate = 16000
            >>> wav_duration = 0.5
            >>> num_channels = 1
            >>> num_frames = sample_rate * wav_duration
            >>> wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1
            >>> waveform = wav_data.tile([num_channels, 1])
            >>> wav_path = "./test.wav"

            >>> current_backend = paddle.audio.backends.get_current_backend()
            >>> print(current_backend)
            wave_backend

            >>> backends = paddle.audio.backends.list_available_backends()
            >>> # default backends is ['wave_backend']
            >>> # backends is ['wave_backend', 'soundfile'], if have installed paddleaudio >= 1.0.2

            >>> if 'soundfile' in backends:
            ...     paddle.audio.backends.set_backend('soundfile')
            ...
            >>> paddle.audio.save(wav_path, waveform, sample_rate)

    """
    if backend_name not in list_available_backends():
        raise NotImplementedError()

    if backend_name == "wave_backend":
        module = wave_backend
    else:
        import paddleaudio

        paddleaudio.backends.set_audio_backend(backend_name)
        module = paddleaudio

    for func in ["save", "load", "info"]:
        setattr(backend, func, getattr(module, func))
        setattr(paddle.audio, func, getattr(module, func))


def _init_set_audio_backend():
    # init the default wave_backend.
    for func in ["save", "load", "info"]:
        setattr(backend, func, getattr(wave_backend, func))
