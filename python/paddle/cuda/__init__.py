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

from paddle import base, core, device as paddle_device
from paddle.device import (
    PaddleStream as Stream,
    _device_to_paddle as _device_to_paddle,
    stream_guard as _PaddleStreamGuard,
)

if TYPE_CHECKING:
    from paddle import CUDAPlace, CustomPlace

    DeviceLike = Union["CUDAPlace", "CustomPlace", int, str, None]


def is_available() -> bool:
    """
    Returns True if CUDA is available and Paddle was built with CUDA support.
    """
    return paddle_device.cuda.device_count() >= 1


def synchronize(device: DeviceLike = None) -> None:
    """
    Args:
        device (int | str | None): Device to synchronize.
            - None: synchronize current device
            - int: device index (e.g., 2 -> 'gpu:2')
            - str: device string (e.g., 'cuda:0' or 'gpu:0')
    """
    dev = _device_to_paddle(device)
    paddle_device.synchronize(dev)


def current_stream(device: DeviceLike = None) -> core.CUDAStream:
    """
    Returns the current stream for the specified device.
    """
    dev = _device_to_paddle(device)
    return paddle_device.current_stream(dev)


def get_device_properties(device: DeviceLike = None):
    """
    Returns the properties of a given device.
    """
    dev = _device_to_paddle(device)
    return paddle_device.cuda.get_device_properties(dev)


def get_device_name(device: DeviceLike = None) -> str:
    """
    Returns the name of a given CUDA device.
    """
    dev = _device_to_paddle(device)
    return paddle_device.cuda.get_device_name(dev)


def get_device_capability(device: DeviceLike = None) -> tuple[int, int]:
    """
    Returns the major and minor compute capability of a given device.
    """
    dev = _device_to_paddle(device)
    return paddle_device.cuda.get_device_capability(dev)


def is_initialized() -> bool:
    return paddle_device.is_compiled_with_cuda()


class StreamContext(_PaddleStreamGuard):
    """
    Stream context manager, inherited from Paddle's stream_guard.
    """

    def __init__(self, stream: paddle_device.Stream):
        super().__init__(stream)


def stream(stream_obj: paddle_device.Stream | None) -> StreamContext:
    """
    A context manager that sets a given stream as the current stream.
    """
    return StreamContext(stream_obj)


def cudart():
    r"""Retrieves the CUDA runtime API module.

    This function initializes the CUDA runtime environment if it is not already
    initialized and returns the CUDA runtime API module (_cudart). The CUDA
    runtime API module provides access to various CUDA runtime functions.

    Args:
        ``None``

    Returns:
        module: The CUDA runtime API module (_cudart).

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> from paddle.cuda import cudart, check_error
            >>> import os
            >>>
            >>> os.environ['CUDA_PROFILE'] = '1'
            >>>
            >>> def perform_cuda_operations_with_streams():
            >>>     stream = paddle.cuda.Stream()
            >>>     with paddle.cuda.stream(stream):
            >>>         x = paddle.randn(100, 100, device='cuda')
            >>>         y = paddle.randn(100, 100, device='cuda')
            >>>         z = paddle.mul(x, y)
            >>>     return z
            >>>
            >>> paddle.cuda.synchronize()
            >>> # print("====== Start nsys profiling ======")
            >>> check_error(cudart().cudaProfilerStart())
            >>> paddle.core.nvprof_start()
            >>> paddle.core.nvprof_nvtx_push("Test")
            >>> result = perform_cuda_operations_with_streams()
            >>> paddle.core.nvprof_nvtx_pop()
            >>> # print("CUDA operations completed.")
            >>> check_error(paddle.cuda.cudart().cudaProfilerStop())
            >>> # print("====== End nsys profiling ======")
    """
    return base.libpaddle._cudart


class CudaError(RuntimeError):
    def __init__(self, code: int) -> None:
        msg = base.libpaddle._cudart.cudaGetErrorString(
            base.libpaddle._cudart.cudaError(code)
        )
        super().__init__(f"{msg} ({code})")


def check_error(res: int) -> None:
    r"""Check the return code of a CUDA runtime API call.

    This function validates whether the given result code from a CUDA
    runtime call indicates success. If the result code is not
    :data:`base.libpaddle._cudart.cudaError.success`, it raises a
    :class:`CudaError`.

    Args:
        res (int): The CUDA runtime return code.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> from paddle.cuda import check_error
            >>> check_error(0) # check for cuda success code # will not raise Error
            >>> # check_error(1) # check for cuda error code 1(invalid argument), will raise Error
            >>> # check_error(2) # check for cuda error code 2(out of memory), will raise Error
    """
    if res != base.libpaddle._cudart.cudaError.success:
        raise CudaError(res)


def mem_get_info(device: DeviceLike = None) -> tuple[int, int]:
    r"""Return the free and total GPU memory (in bytes) for a given device using ``cudaMemGetInfo``.

    This function queries the CUDA runtime for the amount of memory currently
    available and the total memory capacity of the specified device.

    Args:
        device (DeviceLike, optional): The target device. If ``None`` (default),
            the current device, as returned by ``paddle.device.get_device``
            will be used.

    Returns:
        tuple[int, int]: A tuple ``(free, total)``, where
            - ``free`` (int): The number of free bytes of GPU memory available.
            - ``total`` (int): The total number of bytes of GPU memory.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> from paddle.cuda import mem_get_info
            >>> free_bytes, total_bytes = mem_get_info()
    """
    if device is None:
        device: str = paddle_device.get_device()

    if isinstance(device, str):
        device: core.Place = paddle_device._convert_to_place(device)

    if not isinstance(device, core.CUDAPlace) or (
        isinstance(device, core.Place) and not device.is_gpu_place()
    ):
        raise ValueError(f"Expected a cuda device, but got: {device}")

    device_id = (
        device.get_device_id()
        if isinstance(device, core.CUDAPlace)
        else device.gpu_device_id()
    )
    return cudart().cudaMemGetInfo(device_id)


def get_stream_from_external(
    data_ptr: int, device: DeviceLike = None
) -> Stream:
    r"""Return a :class:`paddle.cuda.Stream` from an externally allocated CUDA stream.

    This function is used to wrap streams allocated in other libraries in order
    to facilitate data exchange and multi-library interactions.

    .. note:: This function doesn't manage the stream life-cycle, it is the user
        responsibility to keep the referenced stream alive while this returned
        stream is being used.

    Args:
        data_ptr(int): Integer representation of the `cudaStream_t` value that
            is allocated externally.
        device(paddle.CUDAPlace or int, optional): the device where the stream
            was originally allocated. If device is specified incorrectly,
            subsequent launches using this stream may fail.

    Returns:
        paddle.cuda.Stream: A Stream object wrapping the given external CUDA stream.
    """

    device = _device_to_paddle(device)
    stream_ex = paddle_device.get_stream_from_external(data_ptr, device)

    return stream_ex


__all__ = [
    "cudart",
    "check_error",
    "is_available",
    "is_initialized",
    "mem_get_info",
    "synchronize",
    "current_stream",
    "get_device_properties",
    "get_device_name",
    "get_device_capability",
    "stream",
    "Stream",
    "get_stream_from_external",
]
