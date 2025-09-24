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
from paddle import base, core, device as paddle_device
from paddle.device import (
    PaddleStream as Stream,
    _device_to_paddle as _device_to_paddle,
    manual_seed_all as device_manual_seed_all,
    stream_guard as _PaddleStreamGuard,
)

if TYPE_CHECKING:
    DeviceLike = Union[paddle.core.Place, int, str, None]


def is_available() -> bool:
    """
    Check whether CUDA is available in the current environment

    If Paddle is built with CUDA support and there is at least one CUDA device
    available, this function returns True. Otherwise, it returns False.

    Returns:
        bool: True if CUDA is available, False otherwise.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> if paddle.cuda.is_available():
            ...     print("CUDA is available")
            ... else:
            ...     print("CUDA is not available")
    """
    return paddle_device.cuda.device_count() >= 1


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
    Check whether the current CUDA stream is in capturing state.
    Returns:
        bool: True if current CUDA stream is capturing, False otherwise.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # Check initial state (not capturing)
            >>> print(paddle.cuda.is_current_stream_capturing())  # False

            >>> # Check CUDA availability first
            >>> if paddle.device.device_count()>0:
            ...     # Check initial state (not capturing)
            ...     print(paddle.cuda.is_current_stream_capturing())  # False
            ...
            ...     # Start capturing
            ...     graph = paddle.device.cuda.graphs.CUDAGraph()
            ...     graph.capture_begin()
            ...     print(paddle.cuda.is_current_stream_capturing())  # True
            ...
            ...     # End capturing
            ...     graph.capture_end()
            ...     print(paddle.cuda.is_current_stream_capturing())  # False
    """
    return core.is_cuda_graph_capturing()


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
    dev = _device_to_paddle(device)
    return paddle_device.cuda.get_device_properties(dev)


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
    dev = _device_to_paddle(device)
    return paddle_device.cuda.get_device_name(dev)


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
    dev = _device_to_paddle(device)
    return paddle_device.cuda.get_device_capability(dev)


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


def is_initialized() -> bool:
    return paddle_device.is_compiled_with_cuda()


class StreamContext(_PaddleStreamGuard):
    """
    Notes:
        This API only supports dynamic graph mode currently.
    A context manager that specifies the current stream context by the given stream.

    Args:
        stream(Stream, optional): the selected stream. If stream is None, just yield.

    Returns:
        None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle

            >>> paddle.set_device('cuda')
            >>> s = paddle.cuda.Stream()
            >>> data1 = paddle.ones(shape=[20])
            >>> data2 = paddle.ones(shape=[20])
            >>> data3 = data1 + data2
            >>> with paddle.cuda.StreamContext(s):
            ...     s.wait_stream(paddle.cuda.current_stream()) # type: ignore[attr-defined]
            ...     data4 = data1 + data3

    """

    def __init__(self, stream: paddle_device.Stream):
        super().__init__(stream)


def get_rng_state(device: DeviceLike | None = None) -> core.GeneratorState:
    """
    Return the random number generator state of the specified device as a ByteTensor.

    Args:
        device (DeviceLike, optional): The device to retrieve the RNG state from.
            If not specified, uses the current default device (as returned by paddle.framework._current_expected_place_()).
            Can be a device object, integer device ID, or device string.

    Returns:
        core.GeneratorState: The current RNG state of the specified device, represented as a ByteTensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.cuda.get_rng_state()
    """

    device = _device_to_paddle(device)
    if device is None:
        place = paddle.framework._current_expected_place_()
    else:
        place = paddle_device._convert_to_place(device)
    if isinstance(place, paddle.CPUPlace):
        return core.default_cpu_generator().get_state()
    elif isinstance(place, paddle.CUDAPlace):
        return core.default_cuda_generator(place.get_device_id()).get_state()
    elif isinstance(place, paddle.XPUPlace):
        return core.default_xpu_generator(place.get_device_id()).get_state()
    elif isinstance(place, paddle.CustomPlace):
        return core.default_custom_device_generator(
            paddle.CustomPlace(place.get_device_type(), place.get_device_id())
        ).get_state()


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
    device = _device_to_paddle(device)
    if device is None:
        place = paddle.framework._current_expected_place_()
    else:
        place = paddle_device._convert_to_place(device)

    if isinstance(place, paddle.CUDAPlace):
        core.default_cuda_generator(place.get_device_id()).set_state(new_state)
    elif isinstance(place, paddle.XPUPlace):
        core.default_xpu_generator(place.get_device_id()).set_state(new_state)
    elif isinstance(place, paddle.CustomPlace):
        core.default_custom_device_generator(
            paddle.CustomPlace(place.get_device_type(), place.get_device_id())
        ).set_state(new_state)
    elif isinstance(place, core.CPUPlace):
        core.default_cpu_generator().set_state(new_state)


def stream(stream_obj: paddle_device.Stream | None) -> StreamContext:
    '''

    Notes:
        This API only supports dynamic graph mode currently.
    A context manager that specifies the current stream context by the given stream.

    Args:
        stream(Stream, optional): the selected stream. If stream is None, just yield.

    Returns:
        None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle

            >>> paddle.set_device('cuda')
            >>> s = paddle.cuda.Stream()
            >>> data1 = paddle.ones(shape=[20])
            >>> data2 = paddle.ones(shape=[20])
            >>> data3 = data1 + data2

            >>> with paddle.cuda.stream(s):
            ...     s.wait_stream(paddle.cuda.current_stream())
            ...     data4 = data1 + data3
            >>> print(data4)

    '''
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

            >>> # Wrap it into a Paddle Stream
            >>> # external_stream = paddle.cuda.get_stream_from_external(original_raw_ptr)
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
    "manual_seed_all",
]
