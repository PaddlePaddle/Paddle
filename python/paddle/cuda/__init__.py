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


def current_stream(device: DeviceLike = None) -> paddle.core.CUDAStream:
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
    return paddle.device.cuda.get_device_name(dev)


def get_device_capability(device: DeviceLike = None) -> tuple[int, int]:
    """
    Returns the major and minor compute capability of a given device.
    """
    dev = _device_to_paddle(device)
    return paddle.device.cuda.get_device_capability(dev)


def is_initialized() -> bool:
    return paddle.device.is_compiled_with_cuda()


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


def cudart():
    r"""Retrieves the CUDA runtime API module.

    This function initializes the CUDA runtime environment if it is not already
    initialized and returns the CUDA runtime API module (_cudart). The CUDA
    runtime API module provides access to various CUDA runtime functions.

    Args:
        ``None``

    Returns:
        module: The CUDA runtime API module (_cudart).

    Example of CUDA operations with profiling:
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
        >>> print("====== Start nsys profiling ======")
        >>> check_error(cudart().cudaProfilerStart())
        >>> with paddle.autograd.profiler.emit_nvtx():
        >>>     result = perform_cuda_operations_with_streams()
        >>>     print("CUDA operations completed.")
        >>> check_error(paddle.cuda.cudart().cudaProfilerStop())
        >>> print("====== End nsys profiling ======")

    To run this example and save the profiling information, execute:
        >>> $ nvprof --profile-from-start off --csv --print-summary -o trace_name.prof -f -- python cudart_test.py

    This command profiles the CUDA operations in the provided script and saves
    the profiling information to a file named `trace_name.prof`.
    The `--profile-from-start off` option ensures that profiling starts only
    after the `cudaProfilerStart` call in the script.
    The `--csv` and `--print-summary` options format the profiling output as a
    CSV file and print a summary, respectively.
    The `-o` option specifies the output file name, and the `-f` option forces the
    overwrite of the output file if it already exists.
    """
    return paddle.base.libpaddle._cudart


class CudaError(RuntimeError):
    def __init__(self, code: int) -> None:
        msg = paddle.base.libpaddle._cudart.cudaGetErrorString(
            paddle.base.libpaddle._cudart.cudaError(code)
        )
        super().__init__(f"{msg} ({code})")


def check_error(res: int) -> None:
    if res != paddle.base.libpaddle._cudart.cudaError.success:
        raise CudaError(res)


def mem_get_info(device: DeviceLike | int = None) -> tuple[int, int]:
    r"""Return the global free and total GPU memory for a given device using cudaMemGetInfo.

    Args:
        device (DeviceLike, optional): Selected device. Returns
            statistic for the current device, given by :func:`~paddle.cuda.current_device`,
            if :attr:`device` is ``None`` (default) or if the device index is not specified.

    Returns:
        return
    """
    if device is None:
        device: str = paddle.device.get_device()

    if isinstance(device, str):
        device: paddle.core.Place = paddle.device._convert_to_place(device)

    if not isinstance(device, paddle.core.CUDAPlace) or (
        isinstance(device, paddle.core.Place) and not device.is_gpu_place()
    ):
        raise ValueError(f"Expected a cuda device, but got: {device}")

    device_id = (
        device.get_device_id()
        if isinstance(device, paddle.core.CUDAPlace)
        else device.gpu_device_id()
    )
    return paddle.cuda.cudart().cudaMemGetInfo(device_id)


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
    stream_ex = paddle.device.get_stream_from_external(data_ptr, device)

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
