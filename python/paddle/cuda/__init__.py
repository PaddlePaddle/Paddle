# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# paddle/cuda/__init__.py

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import paddle
from paddle import CUDAPlace, CustomPlace
from paddle.device import (
    Stream as _PaddleStream,
    stream_guard as _PaddleStreamGuard,
)

if TYPE_CHECKING:
    from paddle.base import core

DeviceLike = Union[CUDAPlace, CustomPlace, int, str, None]


def _device_to_paddle(device: DeviceLike) -> str:
    """
    Convert a device spec (int, str, None) to Paddle device string 'gpu:X'.
    Args:
        device: None, int, or str like 'cuda:0' / 'gpu:0'
    Returns:
        str: Paddle device string
    """
    if isinstance(device, (CUDAPlace, CustomPlace)) or device is None:
        return device
    elif isinstance(device, int):
        return f"gpu:{device}"
    elif isinstance(device, str):
        return device.replace("cuda", "gpu")
    else:
        raise TypeError(f"Unsupported device type: {type(device)}")


def is_available() -> bool:
    """
    Mimics torch.cuda.is_available()
    Returns True if CUDA is available and Paddle was built with CUDA support.
    """
    return paddle.device.cuda.device_count() >= 1


def synchronize(device: DeviceLike = None) -> None:
    """
    Mimics torch.cuda.synchronize()
    Args:
        device (int | str | None): Device to synchronize.
            - None: synchronize current device
            - int: device index (e.g., 2 -> 'gpu:2')
            - str: device string (e.g., 'cuda:0' or 'gpu:0')
    """
    dev = _device_to_paddle(device)
    paddle.device.synchronize(dev)


def current_stream(device: DeviceLike = None) -> core.CUDAStream:
    """
    Mimics torch.cuda.current_stream()
    Returns the current stream for the specified device.
    """
    dev = _device_to_paddle(device)
    return paddle.device.current_stream(dev)


def get_device_properties(device: DeviceLike = None):
    """
    Mimics torch.cuda.get_device_properties()
    Returns the properties of a given device.
    """
    dev = _device_to_paddle(device)
    return paddle.device.cuda.get_device_properties(dev)


def get_device_name(device: int | None = None) -> str:
    """
    Mimics torch.cuda.get_device_name()
    Returns the name of a given CUDA device.
    """
    return paddle.device.cuda.get_device_name(device)


def get_device_capability(device: int | None = None) -> tuple[int, int]:
    """
    Mimics torch.cuda.get_device_capability()
    Returns the major and minor compute capability of a given device.
    """
    return paddle.device.cuda.get_device_capability(device)


class StreamContext(_PaddleStreamGuard):
    """
    Torch style Stream context manager, inherited from Paddle's stream_guard.
    """

    def __init__(self, stream: _PaddleStream):
        super().__init__(stream)


def stream(stream_obj: paddle.device.Stream | None) -> StreamContext:
    """
    Mimics torch.cuda.stream()
    A context manager that sets a given stream as the current stream.
    """
    return StreamContext(stream_obj)


class Stream(_PaddleStream):
    """
    Torch API: torch.cuda.Stream -> Paddle: paddle.device.Stream
    """

    # PyTorch priority -> Paddle priority
    _priority_map = {-1: 1, 0: 2}

    def __init__(self, device=None, priority=0, *args, **kwargs):
        """
        Args:
            device (int | str | None): device id/str/None
            priority (int): PyTorch priority (-1, 0)
        """
        paddle_device = _device_to_paddle(device)

        paddle_priority = self._priority_map.get(priority, 2)

        super().__init__(
            device=paddle_device, priority=paddle_priority, *args, **kwargs
        )


__all__ = [
    "is_available",
    "synchronize",
    "current_stream",
    "get_device_properties",
    "get_device_name",
    "get_device_capability",
    "stream",
    "Stream",
]
