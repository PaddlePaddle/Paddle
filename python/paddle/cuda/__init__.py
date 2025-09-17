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
    from paddle import CUDAPlace, CustomPlace, XPUPlace

    DeviceLike = Union["CUDAPlace", "CustomPlace", "XPUPlace", int, str, None]


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
            base.libpaddle._cudart.cudaError_(code)
        )
        super().__init__(f"{msg} ({code})")


def check_error(res: int) -> None:
    r"""Check the return code of a CUDA runtime API call.

    This function validates whether the given result code from a CUDA
    runtime call indicates success. If the result code is not
    :data:`base.libpaddle._cudart.cudaError_.success`, it raises a
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
    if res != base.libpaddle._cudart.cudaError_.success:
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


def current_device() -> int:
    """
    Return the index of a currently selected device.

    Returns:
        int: The index of the currently selected device.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> device_id = paddle.cuda.current_device()
            >>> print(f"Current device index: {device_id}")
    """
    # Use paddle.device.get_device() to get the current device string
    device_str = paddle_device.get_device()

    # Parse the device string to extract the device index
    # Format examples: 'gpu:0', 'xpu:0', 'custom_device:0'
    if ':' in device_str:
        device_id = int(device_str.split(':')[1])
    else:
        # If no device index is specified, default to 0
        device_id = 0

    return device_id


def device_count() -> int:
    """
    Return the number of devices available.

    Returns:
        int: The number of devices available.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> count = paddle.cuda.device_count()
            >>> print(f"Number of devices available: {count}")
    """
    # Use paddle.device.device_count() to get the device count
    # This function supports multiple hardware types (CUDA, XPU, Custom devices)
    return paddle_device.device_count()


def empty_cache() -> None:
    """
    Release all unoccupied cached memory currently held by the caching allocator so that those can be used in other application and visible in nvidia-smi.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> # Create a tensor to allocate memory
            >>> tensor = paddle.randn([1000, 1000], device='cuda')
            >>> # Delete the tensor to free memory (but it may still be cached)
            >>> del tensor
            >>> # Release the cached memory
            >>> paddle.cuda.empty_cache()
    """
    # Use paddle.device.empty_cache() to release cached memory
    # This function supports multiple hardware types (CUDA, XPU, Custom devices)
    paddle_device.empty_cache()


def is_initialized() -> bool:
    """
    Return whether device has been initialized.

    Returns:
        bool: True if any device (CUDA, XPU, or Custom) has been initialized, False otherwise.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> initialized = paddle.cuda.is_initialized()
            >>> print(f"Device initialized: {initialized}")
    """
    # Check if any device type has been compiled/initialized
    # This supports multiple hardware types (CUDA, XPU, Custom devices)
    cuda_initialized = core.is_compiled_with_cuda()
    xpu_initialized = core.is_compiled_with_xpu()

    # Check for custom devices - get all available custom device types
    custom_device_initialized = False
    try:
        custom_device_types = paddle_device.get_all_custom_device_type()
        if custom_device_types:
            # Check if any custom device type is compiled/initialized
            for device_type in custom_device_types:
                if core.is_compiled_with_custom_device(device_type):
                    custom_device_initialized = True
                    break
    except Exception:
        # If there's an error getting custom device types, assume not initialized
        custom_device_initialized = False

    # Return True if any device type is initialized
    return cuda_initialized or xpu_initialized or custom_device_initialized


def memory_allocated(device: DeviceLike = None) -> int:
    """
    Return the current device memory occupied by tensors in bytes for a given device.

    Args:
        device (DeviceLike, optional): The device to query. If None, use the current device.
            Can be paddle.CUDAPlace, paddle.CustomPlace, paddle.XPUPlace, int (device index), or str (device string).

    Returns:
        int: The current memory occupied by tensors in bytes.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> # Get memory allocated for current device
            >>> mem_allocated = paddle.cuda.memory_allocated()
            >>> print(f"Memory allocated: {mem_allocated} bytes")
            >>>
            >>> # Get memory allocated for specific device
            >>> mem_allocated = paddle.cuda.memory_allocated(0)
            >>> print(f"Memory allocated on device 0: {mem_allocated} bytes")
    """
    # Use paddle.device.memory_allocated() to get the memory allocated
    # This function supports multiple hardware types (CUDA, XPU, Custom devices)
    return paddle_device.memory_allocated(device)


def memory_reserved(device: DeviceLike = None) -> int:
    """
    Return the current device memory managed by the caching allocator in bytes for a given device.

    Args:
        device (DeviceLike, optional): The device to query. If None, use the current device.
            Can be paddle.CUDAPlace, paddle.CustomPlace, paddle.XPUPlace, int (device index), or str (device string).

    Returns:
        int: The current memory managed by the caching allocator in bytes.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> import paddle
            >>> # Get memory reserved for current device
            >>> mem_reserved = paddle.cuda.memory_reserved()
            >>> print(f"Memory reserved: {mem_reserved} bytes")
            >>>
            >>> # Get memory reserved for specific device
            >>> mem_reserved = paddle.cuda.memory_reserved(0)
            >>> print(f"Memory reserved on device 0: {mem_reserved} bytes")
    """
    # Use paddle.device.memory_reserved() to get the memory reserved
    # This function supports multiple hardware types (CUDA, XPU, Custom devices)
    return paddle_device.memory_reserved(device)


def set_device(device: DeviceLike) -> None:
    """
    Set the current device.

    Args:
        device (DeviceLike): The device to set as current.
            Can be paddle.CUDAPlace, paddle.CustomPlace, paddle.XPUPlace,
            int (device index), or str (device string).

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> # Set current device to GPU:0
            >>> paddle.cuda.set_device(0)
            >>>
            >>> # Set current device to GPU:1
            >>> paddle.cuda.set_device('gpu:1')
            >>>
            >>> # Set current device to a specific CUDAPlace
            >>> place = paddle.CUDAPlace(0)
            >>> paddle.cuda.set_device(place)
    """
    # Convert device to string format if needed and call paddle.device.set_device()
    # This function supports multiple hardware types (CUDA, XPU, Custom devices)
    if isinstance(device, int):
        # Convert int device index to string format (e.g., 0 -> 'gpu:0')
        device_str = f'gpu:{device}'
    elif isinstance(device, str):
        # Device is already in string format
        device_str = device
    elif isinstance(device, core.CUDAPlace):
        # Convert CUDAPlace object to string format
        device_str = f'gpu:{device.get_device_id()}'
    elif isinstance(device, core.CustomPlace):
        # Convert CustomPlace object to string format
        device_str = f'{device.get_device_type()}:{device.get_device_id()}'
    elif isinstance(device, core.XPUPlace):
        # Convert XPUPlace object to string format
        device_str = f'xpu:{device.get_device_id()}'
    else:
        raise ValueError(
            f"Unsupported device type: {type(device)}. Expected int, str, CUDAPlace, XPUPlace, or CustomPlace."
        )

    # Call paddle.device.set_device() to set the current device
    paddle_device.set_device(device_str)


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
    "current_device",
    "device_count",
    "empty_cache",
    "is_initialized",
    "memory_allocated",
    "memory_reserved",
    "set_device",
]
