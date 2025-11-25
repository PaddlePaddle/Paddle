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
from paddle import base, core, device as paddle_device, framework
from paddle.device import (
    Event,
    Stream,
    StreamContext,
    _device_to_paddle as _device_to_paddle,
    amp,  # noqa: F401
    current_device,
    device,
    ipc_collect,
    is_available as _device_is_available,
    is_bf16_supported,
    is_current_stream_capturing as _is_current_stream_capturing,
    manual_seed,
    manual_seed_all as device_manual_seed_all,
    reset_peak_memory_stats,
    set_stream,
    stream,
)
from paddle.tensor.creation import (
    BFloat16Tensor,
    BoolTensor,
    ByteTensor,
    CharTensor,
    DoubleTensor,
    FloatTensor,
    HalfTensor,
    IntTensor,
    LongTensor,
    ShortTensor,
)

if TYPE_CHECKING:
    DeviceLike = Union[paddle.core.Place, int, str, None]


def is_available() -> bool:
    """
    Check whether **any supported device** is available in the current environment.

    This function checks whether Paddle is built with support for at least one
    type of accelerator (e.g., CUDA, XPU, CustomDevice) and whether there is
    at least one device of that type available.

    If any supported device is available, this function returns True. Otherwise,
    it returns False.

    Returns:
        bool: True if there is at least one available device (GPU/XPU/CustomDevice),
        False otherwise.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> if paddle.cuda.is_available():
            ...     print("At least one device is available")
            ... else:
            ...     print("No supported devices available")
    """
    return _device_is_available()


def synchronize(device: DeviceLike = None) -> None:
    """
    Wait for all streams on a given device to complete.

    This function blocks the calling thread until all the operations
    on the specified device have finished. It is useful for ensuring
    synchronization between CPU and GPU or across multiple devices.

    Args:
        device (CUDAPlace | CustomPlace | int | str | None, optional): The target device to synchronize.
            - None: Synchronize the current device.
            - int: Device index, e.g., ``2`` means ``gpu:2``.
            - str: Device string, e.g., ``'cuda:0'`` or ``'gpu:0'``.
            - CUDAPlace: A Paddle CUDA place object.
            - CustomPlace: A Paddle custom device place object.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle

            # synchronize the current device
            >>> paddle.cuda.synchronize()
    """
    dev = _device_to_paddle(device)
    paddle_device.synchronize(dev)


def current_stream(device: DeviceLike = None) -> Stream:
    """
    Return the current stream for the given device.

    Args:
        device (int | str | paddle.CUDAPlace | paddle.CustomPlace | None, optional):
            The target device to query.

            - None: use the current device.
            - int: device index (e.g., 0 -> 'gpu:0').
            - str: device string (e.g., "cuda:0", "gpu:1").
            - CUDAPlace or CustomPlace: Paddle device objects.

    Returns:
        core.CUDAStream: The current CUDA stream associated with the given device.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle

            # Get the current stream on the default CUDA device
            >>> s1 = paddle.cuda.current_stream()
            >>> print(s1)

            # Get the current stream on device cuda:0
            >>> s2 = paddle.cuda.current_stream("cuda:0")
            >>> print(s2)
    """
    dev = _device_to_paddle(device)
    return paddle_device.current_stream(dev)


def is_current_stream_capturing() -> bool:
    """
    Check whether the current stream is in CUDA graph capturing state.

    Returns:
        bool: True if the current stream is capturing, False otherwise.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> if paddle.device.is_available():
            ...     graph = paddle.device.cuda.graphs.CUDAGraph()
            ...     graph.capture_begin()
            ...     print(paddle.cuda.is_current_stream_capturing())  # True
            ...     graph.capture_end()
    """
    return _is_current_stream_capturing()


def get_device_properties(device: DeviceLike = None):
    """
    Get the properties of a CUDA device.

    Args:
        device (int | str | paddle.CUDAPlace | paddle.CustomPlace | None, optional):
            The target device to query.

            - None: use the current device.
            - int: device index (e.g., 0 -> 'gpu:0').
            - str: device string (e.g., "cuda:0", "gpu:1").
            - CUDAPlace or CustomPlace: Paddle device objects.

    Returns:
        DeviceProperties: An object containing the device properties, such as
        name, total memory, compute capability, and multiprocessor count.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle

            # Get the properties of the current device
            >>> props = paddle.cuda.get_device_properties()
            >>> print(props)

    """
    return paddle_device.get_device_properties(device)


def get_device_name(device: DeviceLike = None) -> str:
    """
    Get the name of a device.

    Args:
        device (int | str | paddle.CUDAPlace | paddle.CustomPlace | None, optional):
            The target device to query.

            - None: use the current device.
            - int: device index (e.g., 0 -> 'gpu:0').
            - str: device string (e.g., "cuda:0", "gpu:1").
            - CUDAPlace or CustomPlace: Paddle device objects.

    Returns:
        str: The name of the CUDA device.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle

            # Get the name of the current CUDA device
            >>> name = paddle.cuda.get_device_name()
            >>> print(name)

            # Get the name of device cuda:0
            >>> name0 = paddle.cuda.get_device_name("cuda:0")
            >>> print(name0)
    """
    return paddle_device.get_device_name(device)


def get_device_capability(device: DeviceLike = None) -> tuple[int, int]:
    """
    Get the compute capability (major, minor) of a device.

    Args:
        device (int | str | paddle.CUDAPlace | paddle.CustomPlace | None, optional):
            The target device to query.

            - None: use the current device.
            - int: device index (e.g., 0 -> 'gpu:0').
            - str: device string (e.g., "cuda:0", "gpu:1").
            - CUDAPlace or CustomPlace: Paddle device objects.

    Returns:
        tuple[int, int]: A tuple ``(major, minor)`` representing the compute capability of the CUDA device.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle

            # Get compute capability of the current CUDA device
            >>> capability = paddle.cuda.get_device_capability()
            >>> print(capability)  # e.g., (8, 0)

            # Get compute capability of device cuda:0
            >>> capability0 = paddle.cuda.get_device_capability("cuda:0")
            >>> print(capability0)
    """
    return paddle_device.get_device_capability(device)


def manual_seed_all(seed: int) -> None:
    """

    Sets the seed for global default generator, which manages the random number generation.

    Args:
        seed(int): The random seed to set.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.cuda.manual_seed_all(102)

    """
    device_manual_seed_all(seed)


def get_rng_state(device: DeviceLike | None = None) -> core.GeneratorState:
    """
    Return the random number generator state of the specified device.

    Args:
        device (DeviceLike, optional): The device to retrieve the RNG state from.
            If not specified, uses the current default device (as returned by paddle.framework._current_expected_place_()).
            Can be a device object, integer device ID, or device string.

    Returns:
        core.GeneratorState: The current RNG state of the specified device.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.cuda.get_rng_state()
    """

    return paddle_device.get_rng_state(device)


def set_rng_state(
    new_state: core.GeneratorState, device: DeviceLike | None = None
) -> None:
    """
    Set the random number generator state of the specified device.

    Args:
        new_state (core.GeneratorState): The desired RNG state to set.
            This should be a state object previously obtained from ``get_rng_state()``.
        device (DeviceLike, optional): The device to set the RNG state for.
            If not specified, uses the current default device (as returned by ``paddle.framework._current_expected_place_()``).
            Can be a device object, integer device ID, or device string.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> # Save RNG state
            >>> state = paddle.cuda.get_rng_state()
            >>> # Do some random operations
            >>> x = paddle.randn([2, 3])
            >>> # Restore RNG state
            >>> paddle.cuda.set_rng_state(state)
    """
    paddle_device.set_rng_state(new_state, device)


class nvtx:
    """Namespace for NVTX marker operations."""

    @staticmethod
    def range_push(msg: str):
        """
        Push an NVTX range marker with the given message.

        Args:
            msg (str): The name of the NVTX range.
        Example:
            .. code-block:: python
                >>> # doctest: +REQUIRES(env:GPU)
                >>> import paddle
                >>> # paddle.device.nvtx.range_push("test") is equivalent to paddle.cuda.nvtx.range_push("test")
                >>> paddle.cuda.nvtx.range_push("test")

        """
        paddle.base.core.nvprof_nvtx_push(msg)

    @staticmethod
    def range_pop():
        """
        Pop the most recent NVTX range marker.
        Example:
            .. code-block:: python
                >>> # doctest: +REQUIRES(env:GPU)
                >>> import paddle
                >>> # paddle.device.nvtx.range_pop("test") is equivalent to paddle.cuda.nvtx.range_pop("test")
                >>> paddle.cuda.nvtx.range_pop()
        """
        paddle.base.core.nvprof_nvtx_pop()


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

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle
            >>> from paddle.cuda import cudart, check_error
            >>> import os
            >>>
            >>> os.environ['CUDA_PROFILE'] = '1'
            >>>
            >>> def perform_cuda_operations_with_streams():
            >>>     stream = paddle.cuda.Stream()
            >>>     with paddle.cuda.stream(stream):
            >>>         x = paddle.randn((100, 100), device='cuda')
            >>>         y = paddle.randn((100, 100), device='cuda')
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

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
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

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> from paddle.cuda import mem_get_info
            >>> free_bytes, total_bytes = mem_get_info()
    """
    if device is None:
        device: str = paddle_device.get_device()

    if isinstance(device, str):
        device: core.Place = paddle_device._convert_to_place(device)

    if isinstance(device, int):
        device_id = device
    else:
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
    custom_device_types = paddle_device.get_all_custom_device_type()
    if custom_device_types:
        # Check if any custom device type is compiled/initialized
        for device_type in custom_device_types:
            if core.is_compiled_with_custom_device(device_type):
                custom_device_initialized = True
                break
    else:
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


def max_memory_allocated(device: DeviceLike = None) -> int:
    '''
    Return the peak size of memory that is allocated to tensor of the given device.

    Note:
        The size of memory allocated to tensor is 256-byte aligned in Paddle, which may larger than the memory size that tensor actually need.
        For instance, a float32 0-D Tensor with shape [] will take up 256 bytes memory, even though storing a float32 data requires only 4 bytes.

    Args:
        device(paddle.CUDAPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    Return:
        int: The peak size of memory that is allocated to tensor of the given device, in bytes.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> max_memory_allocated_size = paddle.cuda.max_memory_allocated(paddle.CUDAPlace(0))
            >>> max_memory_allocated_size = paddle.cuda.max_memory_allocated(0)
            >>> max_memory_allocated_size = paddle.cuda.max_memory_allocated("gpu:0")
    '''
    return paddle_device.max_memory_allocated(device)


def max_memory_reserved(device: DeviceLike = None) -> int:
    '''
    Return the peak size of memory that is held by the allocator of the given device.

    Args:
        device(paddle.Place|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    Return:
        int: The peak size of memory that is held by the allocator of the given device, in bytes.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> max_memory_reserved_size = paddle.cuda.max_memory_reserved(paddle.CUDAPlace(0))
            >>> max_memory_reserved_size = paddle.cuda.max_memory_reserved(0)
            >>> max_memory_reserved_size = paddle.cuda.max_memory_reserved("gpu:0")
    '''
    return paddle_device.max_memory_reserved(device)


def reset_max_memory_allocated(device: DeviceLike | None = None) -> None:
    '''
    Reset the peak size of memory that is allocated to tensor of the given device.

    Args:
        device(paddle.Place|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> paddle.cuda.reset_max_memory_allocated(paddle.CUDAPlace(0))
            >>> paddle.cuda.reset_max_memory_allocated(0)
            >>> paddle.cuda.reset_max_memory_allocated("gpu:0")
    '''

    return paddle_device.reset_max_memory_allocated(device)


def reset_max_memory_reserved(device: DeviceLike | None = None) -> None:
    '''
    Reset the peak size of memory that is held by the allocator of the given device.

    Args:
        device(paddle.Place|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> paddle.cuda.reset_max_memory_reserved(paddle.CUDAPlace(0))
            >>> paddle.cuda.reset_max_memory_reserved(0)
            >>> paddle.cuda.reset_max_memory_reserved("gpu:0")
    '''
    return paddle_device.reset_max_memory_reserved(device)


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

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle
            >>> # Set current device to GPU:0
            >>> paddle.cuda.set_device(0)
            >>> # Set current device to GPU:0
            >>> paddle.cuda.set_device('gpu:0')
            >>> # Set current device to a specific CUDAPlace
            >>> place = paddle.CUDAPlace(0)
            >>> paddle.cuda.set_device(place)
    """
    # Convert device to string format if needed and call paddle.device.set_device()
    # This function supports multiple hardware types (CUDA, XPU, Custom devices)
    if isinstance(device, int):
        # Convert int device index to string format (e.g., 0 -> 'gpu:0')
        device_place = framework._current_expected_place_()
        if isinstance(device_place, core.CUDAPlace):
            device_str = f'gpu:{device}'
        elif isinstance(device_place, core.CustomPlace):
            device_str = f'{device_place.get_device_type()}:{device}'
        elif isinstance(device_place, core.XPUPlace):
            device_str = f'xpu:{device}'
        else:
            raise ValueError(
                "Paddle-CPU is not supported. Please use PaddlePaddle with CUDA, XPU or Custom Device"
            )
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
    """
    Wrap an externally allocated CUDA stream into a Paddle :class:`paddle.cuda.Stream` object.

    This function allows integrating CUDA streams allocated by other libraries
    into Paddle, enabling multi-library interoperability and data exchange.

    Note:
        - This function does not manage the lifetime of the external stream.
          It is the caller's responsibility to ensure the external stream remains valid
          while the returned Paddle stream is in use.
        - Providing an incorrect `device` may result in errors during kernel launches.

    Args:
        data_ptr (int): Integer representation of the external `cudaStream_t`.
        device (DeviceLike, optional): The device where the external stream was created.
            Can be a Paddle device string (e.g., "cuda:0"), an int index (e.g., 0),
            or a PaddlePlace (CUDAPlace). Default: None (current device).

    Returns:
        paddle.cuda.Stream: A Paddle Stream object that wraps the external CUDA stream.

    Examples:
        .. code-block:: python
            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle

            >>> # Assume an external library provides a stream pointer:original_raw_ptr
            >>> # doctest: +SKIP('original_raw_ptr not exist')
            >>> original_raw_ptr = 77777
            >>> external_stream = paddle.cuda.get_stream_from_external(original_raw_ptr)
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
    "set_stream",
    "manual_seed_all",
    "get_rng_state",
    "set_rng_state",
    'FloatTensor',
    'DoubleTensor',
    'HalfTensor',
    'BFloat16Tensor',
    'ByteTensor',
    'CharTensor',
    'ShortTensor',
    'IntTensor',
    'LongTensor',
    'BoolTensor',
    "device",
    "is_bf16_supported",
    "manual_seed",
    "max_memory_allocated",
    "reset_peak_memory_stats",
    "Event",
    "ipc_collect",
    "StreamContext",
]
