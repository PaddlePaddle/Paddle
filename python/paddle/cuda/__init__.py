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
    PaddleStream as Stream,
    _device_to_paddle as _device_to_paddle,
    stream_guard as _PaddleStreamGuard,
)

if TYPE_CHECKING:
    from paddle.base import core

DeviceLike = Union[CUDAPlace, CustomPlace, int, str, None]


def is_available() -> bool:
    """
    Returns True if CUDA is available and Paddle was built with CUDA support.
    """
    return paddle.device.cuda.device_count() >= 1


def synchronize(device: DeviceLike = None) -> None:
    """
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
    Returns the current stream for the specified device.
    """
    dev = _device_to_paddle(device)
    return paddle.device.current_stream(dev)


def get_device_properties(device: DeviceLike = None):
    """
    Returns the properties of a given device.
    """
    dev = _device_to_paddle(device)
    return paddle.device.cuda.get_device_properties(dev)


def get_device_name(device: DeviceLike = None) -> str:
    """
    Returns the name of a given CUDA device.
    """
    dev = _device_to_paddle(device)
    return paddle.device.cuda.get_device_name(device)


def get_device_capability(device: DeviceLike = None) -> tuple[int, int]:
    """
    Returns the major and minor compute capability of a given device.
    """
    dev = _device_to_paddle(device)
    return paddle.device.cuda.get_device_capability(device)


class StreamContext(_PaddleStreamGuard):
    """
    Stream context manager, inherited from Paddle's stream_guard.
    """

    def __init__(self, stream: paddle.device.Stream):
        super().__init__(stream)


def stream(stream_obj: paddle.device.Stream | None) -> StreamContext:
    """
    A context manager that sets a given stream as the current stream.
    """
    return StreamContext(stream_obj)


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
