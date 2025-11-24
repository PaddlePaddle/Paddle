# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn, Union

from typing_extensions import TypeAlias

import paddle
from paddle.base import core
from paddle.base.wrapped_decorator import signature_safe_contextmanager
from paddle.utils import deprecated

from .streams import Event, Stream, create_event, create_stream  # noqa: F401

if TYPE_CHECKING:
    from paddle import CUDAPlace, CustomPlace
    from paddle.base.libpaddle import _gpuDeviceProperties

    _CudaPlaceLike: TypeAlias = Union[
        CUDAPlace,
        CustomPlace,
        str,  # some string like "gpu:0", "custom_device:0", etc.
        int,  # some int like 0, 1, etc.
    ]
__all__ = [
    'Stream',
    'Event',
    'current_stream',
    'synchronize',
    'device_count',
    'empty_cache',
    'max_memory_allocated',
    'max_memory_reserved',
    'memory_allocated',
    'memory_reserved',
    'stream_guard',
    'get_device_properties',
    'get_device_name',
    'get_device_capability',
    'reset_max_memory_allocated',
    'reset_max_memory_reserved',
    'memory_summary',
    'vmm_max_free_size',
    'vmm_compact',
    'vmm_free_block_info',
    'vmm_all_block_info',
]


@deprecated(
    since="2.5.0",
    update_to="paddle.device.current_stream",
    level=1,
    reason="current_stream in paddle.device.cuda will be removed in future",
)
def current_stream(device: _CudaPlaceLike | None = None) -> core.CUDAStream:
    '''
    Return the current CUDA stream by the device.

    Args:
        device(paddle.CUDAPlace()|int|None, optional): The device or the ID of the device which want to get stream from.
                If device is None, the device is the current device. Default: None.

    Returns:
            CUDAStream: the stream to the device.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')

            >>> s1 = paddle.device.cuda.current_stream()

            >>> s2 = paddle.device.cuda.current_stream(0)

            >>> s3 = paddle.device.cuda.current_stream(paddle.CUDAPlace(0))

    '''

    device_id = -1

    if device is not None:
        if isinstance(device, int):
            device_id = device
        elif isinstance(device, core.CUDAPlace):
            device_id = device.get_device_id()
        elif isinstance(device, str):
            place = paddle.device._convert_to_place(device)
            device_id = place.get_device_id()
        else:
            raise ValueError("device type must be int or paddle.CUDAPlace")

    return core._get_current_stream(device_id)


@deprecated(
    since="2.5.0",
    update_to="paddle.device.synchronize",
    level=1,
    reason="synchronize in paddle.device.cuda will be removed in future",
)
def synchronize(device: _CudaPlaceLike | None = None) -> None:
    '''
    Wait for the compute on the given CUDA device to finish.

    Args:
        device(paddle.CUDAPlace()|int|None, optional): The device or the ID of the device.
                If device is None, the device is the current device. Default: None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle

            >>> paddle.device.cuda.synchronize()
            >>> paddle.device.cuda.synchronize(0)
            >>> paddle.device.cuda.synchronize(paddle.CUDAPlace(0))

    '''
    if device is not None:
        if isinstance(device, int):
            device_id = device
        elif isinstance(device, core.CUDAPlace):
            device_id = device.get_device_id()
        elif isinstance(device, str):
            if device.startswith('gpu:'):
                device_id = int(device[4:])
            elif device == 'gpu':
                device_id = 0
            else:
                raise ValueError(
                    f"The current string {device} is not expected. Because paddle.device.cuda."
                    "synchronize only support string which is like 'gpu:x' or 'gpu'. "
                    "Please input appropriate string again!"
                )
        else:
            raise ValueError("device type must be int, str or paddle.CUDAPlace")
    else:
        place = paddle.framework._current_expected_place()
        if paddle.is_compiled_with_cuda() and isinstance(
            place, paddle.CUDAPlace
        ):
            device_id = place.get_device_id()
        else:
            device_id = -1
    return core._device_synchronize(device_id)


def device_count() -> int:
    '''
    Return the number of GPUs available.

    Returns:
        int: the number of GPUs available.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.device.cuda.device_count()

    '''

    num_gpus = (
        core.get_cuda_device_count()
        if hasattr(core, 'get_cuda_device_count')
        else 0
    )

    return num_gpus


def empty_cache() -> None:
    '''
    Releases idle cached memory held by the allocator so that those can be used in other GPU
    application and visible in `nvidia-smi`. In most cases you don't need to use this function,
    Paddle does not release the memory back to the OS when you remove Tensors on the GPU,
    Because it keeps gpu memory in a pool so that next allocations can be done much faster.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')

            >>> tensor = paddle.randn([512, 512, 512], "float64")
            >>> del tensor
            >>> paddle.device.cuda.empty_cache()
    '''

    if core.is_compiled_with_cuda():
        core.cuda_empty_cache()


def extract_cuda_device_id(device: _CudaPlaceLike, op_name: str) -> int:
    '''
    Return the id of the given device. It is just a utility that will not be exposed to users.

    Args:
        device(paddle.CUDAPlace|paddle.CustomPlace|int|str): The device, the id of the device or
            the string name of device like 'gpu:x' or 'custom_device:x'.
            Default: None.

    Return:
        int: The id of the given device. If device is None, return the id of current device.
    '''
    if device is None:
        return core.get_cuda_current_device_id()

    if isinstance(device, int):
        device_id = device
        if core.is_compiled_with_cuda():
            device_type = 'gpu'
        else:
            device_type = None
            available_custom_devices = core.get_available_custom_device()
            if len(available_custom_devices) == 1:
                if device == 0:
                    device_type = available_custom_devices[0]
                else:
                    raise ValueError(
                        f"Device id {device} not found in available_custom_devices: [{available_custom_devices[0]}:0]"
                    )
            else:
                for d in available_custom_devices:
                    dev_type, dev_id = d.split(':')
                    if int(dev_id) == device:
                        device_type = dev_type
            if device_type is None:
                raise ValueError(
                    f"Device id {device} not found in available_custom_devices: {available_custom_devices}"
                )
    elif isinstance(device, core.CUDAPlace):
        device_type = 'gpu'
        device_id = device.get_device_id()
    elif isinstance(device, core.CustomPlace):
        device_type = device.get_device_type()
        device_id = device.get_device_id()
    elif isinstance(device, str):
        if device.startswith('gpu:'):
            device_id = int(device[4:])
        elif (
            ':' in device
        ):  # handle custom device formats like npu:0, metax_gpu:1
            device_type, device_id_str = device.split(':', 1)
            device_id = int(device_id_str)
        else:
            raise ValueError(
                f"The current string {device} is not expected. Because {op_name} only support string which is like 'gpu:x' or '<custom_device>:x'. "
                "Please input appropriate string again!"
            )
    else:
        raise ValueError(
            f"The device type {device} is not expected. Because {op_name} only support int, str (format 'gpu:x' or '<custom_device>:x'), paddle.CUDAPlace or paddle.CustomPlace. "
            "Please input appropriate device again!"
        )

    assert device_id >= 0, (
        f"The device id must be not less than 0, but got id = {device_id}."
    )

    if core.is_compiled_with_cuda():
        assert device_id < device_count(), (
            f"The device id {device_id} exceeds gpu card number {device_count()}"
        )
    else:
        assert device_id < core.get_custom_device_count(device_type), (
            f"The device id {device_id} exceeds {device_type} device card number {core.get_custom_device_count(device_type)}"
        )
    return device_id


def max_memory_allocated(device: _CudaPlaceLike | None = None) -> int:
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

            >>> max_memory_allocated_size = paddle.device.cuda.max_memory_allocated(paddle.CUDAPlace(0))
            >>> max_memory_allocated_size = paddle.device.cuda.max_memory_allocated(0)
            >>> max_memory_allocated_size = paddle.device.cuda.max_memory_allocated("gpu:0")
    '''
    name = "paddle.device.cuda.max_memory_allocated"
    custom_devices = paddle.device.get_all_custom_device_type()
    if not (
        core.is_compiled_with_cuda()
        or (
            custom_devices
            and core.is_compiled_with_custom_device(custom_devices[0])
        )
    ):
        raise ValueError(
            f"The API {name} is not supported in CPU-only PaddlePaddle. Please reinstall PaddlePaddle with GPU or custom device support to call this API."
        )
    device_id = extract_cuda_device_id(device, op_name=name)
    return core.device_memory_stat_peak_value("Allocated", device_id)


def max_memory_reserved(device: _CudaPlaceLike | None = None) -> int:
    '''
    Return the peak size of memory that is held by the allocator of the given device.

    Args:
        device(paddle.CUDAPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    Return:
        int: The peak size of memory that is held by the allocator of the given device, in bytes.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> max_memory_reserved_size = paddle.device.cuda.max_memory_reserved(paddle.CUDAPlace(0))
            >>> max_memory_reserved_size = paddle.device.cuda.max_memory_reserved(0)
            >>> max_memory_reserved_size = paddle.device.cuda.max_memory_reserved("gpu:0")
    '''
    name = "paddle.device.cuda.max_memory_reserved"
    custom_devices = paddle.device.get_all_custom_device_type()
    if not (
        core.is_compiled_with_cuda()
        or (
            custom_devices
            and core.is_compiled_with_custom_device(custom_devices[0])
        )
    ):
        raise ValueError(
            f"The API {name} is not supported in CPU-only PaddlePaddle. Please reinstall PaddlePaddle with GPU or custom device support to call this API."
        )
    device_id = extract_cuda_device_id(device, op_name=name)
    return core.device_memory_stat_peak_value("Reserved", device_id)


def reset_max_memory_allocated(device: _CudaPlaceLike | None = None) -> None:
    '''
    Reset the peak size of memory that is allocated to tensor of the given device.

    Args:
        device(paddle.CUDAPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> paddle.device.cuda.reset_max_memory_allocated(paddle.CUDAPlace(0))
            >>> paddle.device.cuda.reset_max_memory_allocated(0)
            >>> paddle.device.cuda.reset_max_memory_allocated("gpu:0")
    '''

    name = "paddle.device.cuda.reset_max_memory_allocated"
    custom_devices = paddle.device.get_all_custom_device_type()
    if not (
        core.is_compiled_with_cuda()
        or (
            custom_devices
            and core.is_compiled_with_custom_device(custom_devices[0])
        )
    ):
        raise ValueError(
            f"The API {name} is not supported in CPU-only PaddlePaddle. Please reinstall PaddlePaddle with GPU or custom device support to call this API."
        )
    device_id = extract_cuda_device_id(device, op_name=name)
    core.device_memory_stat_reset_peak_value("Allocated", device_id)


def reset_max_memory_reserved(device: _CudaPlaceLike | None = None) -> None:
    '''
    Reset the peak size of memory that is held by the allocator of the given device.

    Args:
        device(paddle.CUDAPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> paddle.device.cuda.reset_max_memory_reserved(paddle.CUDAPlace(0))
            >>> paddle.device.cuda.reset_max_memory_reserved(0)
            >>> paddle.device.cuda.reset_max_memory_reserved("gpu:0")
    '''

    name = "paddle.device.cuda.reset_max_memory_reserved"
    custom_devices = paddle.device.get_all_custom_device_type()
    if not (
        core.is_compiled_with_cuda()
        or (
            custom_devices
            and core.is_compiled_with_custom_device(custom_devices[0])
        )
    ):
        raise ValueError(
            f"The API {name} is not supported in CPU-only PaddlePaddle. Please reinstall PaddlePaddle with GPU or custom device support to call this API."
        )
    device_id = extract_cuda_device_id(device, op_name=name)
    core.device_memory_stat_reset_peak_value("Reserved", device_id)


def memory_allocated(device: _CudaPlaceLike | None = None) -> int:
    '''
    Return the current size of memory that is allocated to tensor of the given device.

    Note:
        The size of memory allocated to tensor is 256-byte aligned in Paddle, which may be larger than the memory size that tensor actually need.
        For instance, a float32 0-D Tensor with shape [] will take up 256 bytes memory, even though storing a float32 data requires only 4 bytes.

    Args:
        device(paddle.CUDAPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    Return:
        int: The current size of memory that is allocated to tensor of the given device, in bytes.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> memory_allocated_size = paddle.device.cuda.memory_allocated(paddle.CUDAPlace(0))
            >>> memory_allocated_size = paddle.device.cuda.memory_allocated(0)
            >>> memory_allocated_size = paddle.device.cuda.memory_allocated("gpu:0")
    '''
    name = "paddle.device.cuda.memory_allocated"
    custom_devices = paddle.device.get_all_custom_device_type()
    if not (
        core.is_compiled_with_cuda()
        or (
            custom_devices
            and core.is_compiled_with_custom_device(custom_devices[0])
        )
    ):
        raise ValueError(
            f"The API {name} is not supported in CPU-only PaddlePaddle. Please reinstall PaddlePaddle with GPU or custom device support to call this API."
        )
    device_id = extract_cuda_device_id(device, op_name=name)
    return core.device_memory_stat_current_value("Allocated", device_id)


def memory_reserved(device: _CudaPlaceLike | None = None) -> int:
    '''
    Return the current size of memory that is held by the allocator of the given device.

    Args:
        device(paddle.CUDAPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    Return:
        int: The current size of memory that is held by the allocator of the given device, in bytes.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> memory_reserved_size = paddle.device.cuda.memory_reserved(paddle.CUDAPlace(0))
            >>> memory_reserved_size = paddle.device.cuda.memory_reserved(0)
            >>> memory_reserved_size = paddle.device.cuda.memory_reserved("gpu:0")
    '''
    name = "paddle.device.cuda.memory_reserved"
    custom_devices = paddle.device.get_all_custom_device_type()
    if not (
        core.is_compiled_with_cuda()
        or (
            custom_devices
            and core.is_compiled_with_custom_device(custom_devices[0])
        )
    ):
        raise ValueError(
            f"The API {name} is not supported in CPU-only PaddlePaddle. Please reinstall PaddlePaddle with GPU or custom device support to call this API."
        )
    device_id = extract_cuda_device_id(device, op_name=name)
    return core.device_memory_stat_current_value("Reserved", device_id)


def _set_current_stream(stream: Stream) -> core.CUDAStream:
    '''
    Set the current stream.

    Parameters:
        stream(paddle.device.cuda.Stream): The selected stream.

    Returns:
        CUDAStream: The previous stream.

    '''

    if not isinstance(stream, paddle.device.cuda.Stream):
        raise TypeError("stream type should be paddle.device.cuda.Stream")

    cur_stream = current_stream()
    if id(stream) == id(cur_stream):
        return stream
    return core._set_current_stream(stream)


@deprecated(
    since="2.5.0",
    update_to="paddle.device.stream_guard",
    level=1,
    reason="stream_guard in paddle.device.cuda will be removed in future",
)
@signature_safe_contextmanager
def stream_guard(stream: Stream) -> NoReturn:
    '''
    Notes:
        This API only supports dynamic graph mode currently.

    A context manager that specifies the current stream context by the given stream.

    Parameters:
        stream(paddle.device.cuda.Stream): the selected stream. If stream is None, just yield.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')

            >>> s = paddle.device.cuda.Stream()
            >>> data1 = paddle.ones(shape=[20])
            >>> data2 = paddle.ones(shape=[20])
            >>> with paddle.device.cuda.stream_guard(s):
            ...     data3 = data1 + data2

    '''

    if stream is not None and not isinstance(stream, paddle.device.cuda.Stream):
        raise TypeError("stream type should be paddle.device.cuda.Stream")

    cur_stream = current_stream()
    if stream is None or id(stream) == id(cur_stream):
        yield
    else:
        pre_stream = _set_current_stream(stream)
        try:
            yield
        finally:
            stream = _set_current_stream(pre_stream)


def get_device_properties(
    device: _CudaPlaceLike | None = None,
) -> _gpuDeviceProperties:
    '''
    Return the properties of given device.

    Args:
        device(paddle.CUDAPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x' which to get the properties of the
            device from. If device is None, the device is the current device.
            Default: None.

    Returns:
        _gpuDeviceProperties: The properties of the device which include ASCII string
        identifying device, major compute capability, minor compute capability, global
        memory available and the number of multiprocessors on the device.

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)

            >>> import paddle
            >>> paddle.device.set_device('gpu')
            >>> paddle.device.cuda.get_device_properties()
            >>> # _gpuDeviceProperties(name='A100-SXM4-40GB', major=8, minor=0, total_memory=40536MB, multi_processor_count=108)

            >>> paddle.device.cuda.get_device_properties(0)
            >>> # _gpuDeviceProperties(name='A100-SXM4-40GB', major=8, minor=0, total_memory=40536MB, multi_processor_count=108)

            >>> paddle.device.cuda.get_device_properties('gpu:0')
            >>> # _gpuDeviceProperties(name='A100-SXM4-40GB', major=8, minor=0, total_memory=40536MB, multi_processor_count=108)

            >>> paddle.device.cuda.get_device_properties(paddle.CUDAPlace(0))
            >>> # _gpuDeviceProperties(name='A100-SXM4-40GB', major=8, minor=0, total_memory=40536MB, multi_processor_count=108)

    '''

    if not core.is_compiled_with_cuda():
        raise ValueError(
            "The API paddle.device.cuda.get_device_properties is not supported in "
            "CPU-only PaddlePaddle. Please reinstall PaddlePaddle with GPU support "
            "to call this API."
        )

    if device is not None:
        if isinstance(device, int):
            device_id = device
        elif isinstance(device, core.CUDAPlace):
            device_id = device.get_device_id()
        elif isinstance(device, str):
            if device.startswith('gpu:'):
                device_id = int(device[4:])
            elif device == 'gpu':
                device_id = 0
            else:
                raise ValueError(
                    f"The current string {device} is not expected. Because paddle.device."
                    "cuda.get_device_properties only support string which is like 'gpu:x' or 'gpu'. "
                    "Please input appropriate string again!"
                )
        else:
            raise ValueError(
                f"The device type {device} is not expected. Because paddle.device.cuda."
                "get_device_properties only support int, str or paddle.CUDAPlace. "
                "Please input appropriate device again!"
            )
    else:
        device_id = -1

    return core.get_device_properties(device_id)


def get_device_name(device: _CudaPlaceLike | None = None) -> str:
    '''
    Return the name of the device which is got from CUDA function `cudaDeviceProp <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0>`_.

    Parameters:
        device(paddle.CUDAPlace|int|None, optional): The device or the ID of the device. If device is None (default), the device is the current device.

    Returns:
        str: The name of the device.

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')

            >>> paddle.device.cuda.get_device_name()

            >>> paddle.device.cuda.get_device_name(0)

            >>> paddle.device.cuda.get_device_name(paddle.CUDAPlace(0))

    '''

    return get_device_properties(device).name


def get_device_capability(
    device: _CudaPlaceLike | None = None,
) -> tuple[int, int]:
    """
    Return the major and minor revision numbers defining the device's compute capability which are got from CUDA function `cudaDeviceProp <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0>`_.

    Parameters:
        device(paddle.CUDAPlace|int|None, optional): The device or the ID of the device. If device is None (default), the device is the current device.

    Returns:
        tuple(int,int): the major and minor revision numbers defining the device's compute capability.

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)

            >>> import paddle
            >>> paddle.device.set_device('gpu')
            >>> paddle.device.cuda.get_device_capability()

            >>> paddle.device.cuda.get_device_capability(0)

            >>> paddle.device.cuda.get_device_capability(paddle.CUDAPlace(0))

    """
    prop = get_device_properties(device)
    return prop.major, prop.minor


def get_rng_state(device: _CudaPlaceLike | None = None) -> core.GeneratorState:
    r'''
    Get the random state for the default generator.

    Returns:
        Tensor: The random state tensor.

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.get_rng_state()

    '''
    place = paddle.device.device_to_place(device)
    if isinstance(place, core.CPUPlace):
        return core.default_cpu_generator().get_state()
    return core.default_cuda_generator(place.get_device_id()).get_state()


def set_rng_state(
    new_state: core.GeneratorState, device: _CudaPlaceLike | None = None
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

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> # Save RNG state
            >>> state = paddle.device.get_rng_state()
            >>> # Do some random operations
            >>> x = paddle.randn([2, 3])
            >>> # Restore RNG state
            >>> paddle.device.set_rng_state(state)
    """
    place = paddle.device.device_to_place(device)
    if isinstance(place, core.CPUPlace):
        core.default_cpu_generator().set_state(new_state)
    else:
        core.default_cuda_generator(place.get_device_id()).set_state(new_state)


def manual_seed(seed: int) -> None:
    """Set the seed for generating random numbers for the current Device.

    .. warning::
        If you are working with a multi-Device model, this function is insufficient
        to get determinism.  To seed all Devices, use :func:`manual_seed_all`.
        If current Device is CPU, this function will set the seed of the default CPU generator.

    Sets the seed for global default generator, which manages the random number generation.

    Args:
        seed(int): The random seed to set.

    Returns:
        None

    Examples:
        .. code-block:: python
            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle
            >>> paddle.device.manual_seed(102)
            >>> # paddle.cuda.manual_seed(102) is equivalent to paddle.device.manual_seed(102)
            >>> paddle.cuda.manual_seed(102)

    """
    seed = int(seed)
    place = paddle.framework._current_expected_place_()
    if isinstance(place, core.CPUPlace):
        core.default_cpu_generator().manual_seed(seed)
    else:
        core.default_cuda_generator(place.get_device_id()).manual_seed(seed)


def vmm_max_free_size(device: _CudaPlaceLike | None = None) -> tuple[int, int]:
    '''
    Return the largest continuous free memory block size and the total free size
    managed by the Virtual Memory Management (VMM) allocator of the given device.

    Args:
        device(paddle.CUDAPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    Returns:
        tuple[int, int]: A tuple containing the largest continuous free memory block size (in bytes)
        and the total free memory size (in bytes).

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> max_free, total_free = paddle.device.cuda.vmm_max_free_size(0)
            >>> print(f"Max free size: {max_free}, Total free size: {total_free}")
    '''
    name = 'paddle.device.cuda.vmm_max_free_size'
    if not (core.is_compiled_with_cuda()):
        raise ValueError(
            f"The API {name} is not supported in CPU-only PaddlePaddle. Please reinstall PaddlePaddle with GPU support to call this API."
        )
    device_id = extract_cuda_device_id(device, op_name=name)
    return core.vmm_max_free_size(device_id)


def vmm_compact(device: _CudaPlaceLike | None = None) -> int:
    '''
    Defragment the free memory blocks managed by the Virtual Memory Management (VMM)
    allocator of the given device.

    Args:
        device(paddle.CUDAPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    Returns:
        int: The amount of memory (in bytes) that was moved during the compaction.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> moved_bytes = paddle.device.cuda.vmm_compact(0)
            >>> print(f"Bytes moved during compaction: {moved_bytes}")
    '''
    name = 'paddle.device.cuda.vmm_compact'
    if not (core.is_compiled_with_cuda()):
        raise ValueError(
            f"The API {name} is not supported in CPU-only PaddlePaddle. Please reinstall PaddlePaddle with GPU support to call this API."
        )
    device_id = extract_cuda_device_id(device, op_name=name)
    return core.vmm_compact(device_id)


def vmm_free_block_info(
    device: _CudaPlaceLike | None = None,
) -> list[list[tuple[int, int]]]:
    '''
    Return detailed information about all free memory blocks managed by the Virtual Memory Management (VMM)
    allocator of the given device.

    Args:
        device(paddle.CUDAPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    Returns:
        list[list[tuple[int, int]]]: A nested list. The outer list corresponds to different
        Allocator. The inner list contains tuples, where each tuple is (size_in_bytes, allocation_ptr).

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> info = paddle.device.cuda.vmm_free_block_info(0)
            >>> # info might look like: [[(2002049024, 43983227392)], [(3002069522, 46983227392)]]
            >>> print(info)
    '''
    name = 'paddle.device.cuda.vmm_free_block_info'
    if not (core.is_compiled_with_cuda()):
        raise ValueError(
            f"The API {name} is not supported in CPU-only PaddlePaddle. Please reinstall PaddlePaddle with GPU support to call this API."
        )
    device_id = extract_cuda_device_id(device, op_name=name)
    return core.vmm_free_block_info(device_id)


def vmm_all_block_info(
    device: _CudaPlaceLike | None = None,
) -> list[list[tuple[int, int, bool]]]:
    '''
    Return detailed information about all memory blocks (both free and allocated) managed by
    the Virtual Memory Management (VMM) allocator of the given device.

    Args:
        device(paddle.CUDAPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    Returns:
        list[list[tuple[int, int, bool]]]: A nested list. The outer list corresponds to different
        Allocator. The inner list contains tuples, where each
        tuple is (size_in_bytes, allocation_ptr, is_free).

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> info = paddle.device.cuda.vmm_all_block_info(0)
            >>> # info might look like: [[(2002049024, 43983227392, True), (3002069522, 46983227392, False)]]
            >>> print(info)
    '''
    name = 'paddle.device.cuda.vmm_all_block_info'
    if not (core.is_compiled_with_cuda()):
        raise ValueError(
            f"The API {name} is not supported in CPU-only PaddlePaddle. Please reinstall PaddlePaddle with GPU support to call this API."
        )
    device_id = extract_cuda_device_id(device, op_name=name)
    return core.vmm_all_block_info(device_id)


def memory_summary(device: _CudaPlaceLike | None = None) -> None:
    '''
    Return a string containing a detailed summary of the CUDA memory usage
    for the specified device, printed in three distinct sections: Global Summary,
    Allocator Summary, and Distribution. This function prints the summary directly
    to the terminal.

    Args:
        device(paddle.CUDAPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    The summary includes:
    1. Global Summary: GPU utilization rates and physical memory information (similar to nvidia-smi).
    2. Allocator Summary: Memory allocated by the PaddlePaddle's allocator (Total, Used, Free),
       including a Weighted Fragmentation Rate.
    3. Distribution: A wide pivot table showing the size distribution of allocated blocks
       (split by common sizes like 1M, 10M, ... 3G).

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> paddle.device.cuda.memory_summary(0)
    '''
    nvidia_smi_AVAILABLE = False
    try:
        # import nvidia_smi, pip install nvidia-ml-py3
        import nvidia_smi

        nvidia_smi_AVAILABLE = True
    except ImportError:
        nvidia_smi_AVAILABLE = False

    # --- Constants ---
    KB = 1024
    MB = 1024 * 1024
    GB = 1024 * 1024 * 1024

    THRESHOLDS = [
        1 * MB,
        10 * MB,
        50 * MB,
        100 * MB,
        200 * MB,
        400 * MB,
        600 * MB,
        800 * MB,
        1 * GB,
        2 * GB,
        3 * GB,
    ]
    RANGE_HEADERS = [
        "[0B,1M)",
        "[1M,10M)",
        "[10M,50M)",
        "[50M,100M)",
        "[100M,200M)",
        "[200M,400M)",
        "[400M,600M)",
        "[600M,800M)",
        "[800M,1G)",
        "[1G,2G)",
        "[2G,3G)",
        "[3G,+INF)",
    ]

    allocator_lists = vmm_all_block_info(device=device)

    # --- Formatting Helpers ---
    def format_size(size_bytes):
        if size_bytes == 0:
            return "0 B"
        if size_bytes < MB:
            return f"{size_bytes / KB:.2f} KB"
        if size_bytes < GB:
            return f"{size_bytes / MB:.2f} MB"
        return f"{size_bytes / GB:.2f} GB"

    def print_table(title, headers, rows):
        if not rows:
            return
        # Calculate widths
        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        col_widths = [w + 2 for w in col_widths]

        # Build lines
        row_fmt = "|" + "|".join([f"{{:^{w}}}" for w in col_widths]) + "|"
        header_sep = "+" + "+".join(["=" * w for w in col_widths]) + "+"
        inner_sep = "+" + "+".join(["-" * w for w in col_widths]) + "+"

        print(f"\n### {title}")
        print(header_sep)
        print(
            "|"
            + "|".join([f"{h:^{w}}" for h, w in zip(headers, col_widths)])
            + "|"
        )
        print(header_sep)

        for i, row in enumerate(rows):
            print(row_fmt.format(*[str(c) for c in row]))
            if (
                title == "Block Size Distribution"
                and (i + 1) % 2 == 0
                and i != len(rows) - 1
            ):
                print(inner_sep)
            elif title != "Block Size Distribution":
                print(inner_sep)
        if title == "Block Size Distribution":
            print(header_sep)

    # --- Feature 1: Global Summary with NVML & Rates ---

    # 1.1 Get Paddle Stats
    mem_allocated = paddle.device.cuda.memory_allocated()
    max_mem_allocated = paddle.device.cuda.max_memory_allocated()
    mem_reserved = paddle.device.cuda.memory_reserved()
    max_mem_reserved = paddle.device.cuda.max_memory_reserved()

    # 1.2 Calculate Rates (Utilization of the Reserved Pool)
    # Rate = How much of the reserved pool is actually holding tensor data?
    cur_alloc_rate = (
        ((mem_reserved - mem_allocated) / mem_reserved)
        if mem_reserved > 0
        else 0.0
    )
    max_alloc_rate = (
        ((mem_reserved - max_mem_allocated) / mem_reserved)
        if mem_reserved > 0
        else 0.0
    )

    # 1.3 Get Physical Usage via nvidia_smi
    phy_used_str = "N/A"
    if nvidia_smi_AVAILABLE:
        try:
            nvidia_smi.nvmlInit()
            device_id = extract_cuda_device_id(
                device, op_name="paddle.device.cuda.memory_summary"
            )
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            phy_used_str = format_size(info.used)
            phy_total_str = format_size(info.total)
            # nvidia_smi.nvmlShutdown() # Optional, depends on lifecycle
        except Exception as e:
            phy_used_str = "Err"
            phy_total_str = "Err"
    else:
        print(
            "Place install nvidia-smi to check real memory usage, pip install command: `pip install nvidia-ml-py3`"
        )
        phy_used_str = "No nvidia_smi"
        phy_total_str = "No nvidia_smi"

    global_headers = [
        "Allocators",
        "Allocated",
        "Max Alloc",
        "Reserved",
        "Max Reserved",
        "Cur Util Rate",
        "Max Util Rate",
        "Phy GPU Used / Total",
    ]

    global_rows = [
        [
            len(allocator_lists),
            format_size(mem_allocated),
            format_size(max_mem_allocated),
            format_size(mem_reserved),
            format_size(max_mem_reserved),
            f"{cur_alloc_rate:.2%}",
            f"{max_alloc_rate:.2%}",
            phy_used_str + ' / ' + phy_total_str,
        ]
    ]

    print_table("Global Memory Snapshot", global_headers, global_rows)

    # --- 2. Allocator Analysis ---
    summary_rows = []
    dist_rows = []

    for idx, blocks in enumerate(allocator_lists):
        allocator_name = f"Allocator_{idx}"

        # A. Basic Counting
        total_blocks = len(blocks)
        free_blocks = 0
        total_size = 0
        free_size = 0
        max_free_size = 0
        max_used_size = 0
        buckets = [[0, 0] for _ in range(len(RANGE_HEADERS))]

        for size, addr, is_free in blocks:
            total_size += size
            if is_free:
                free_blocks += 1
                free_size += size
                max_free_size = max(max_free_size, size)
            else:
                max_used_size = max(max_used_size, size)

            # Bucket Mapping
            b_idx = len(THRESHOLDS)
            for i, t in enumerate(THRESHOLDS):
                if size < t:
                    b_idx = i
                    break
            buckets[b_idx][0 if is_free else 1] += 1

        used_blocks = total_blocks - free_blocks
        used_size = total_size - free_size

        # B. Advanced Fragmentation Calculation
        frag_ratio = 0.0

        if free_size > 0 and total_blocks > 0:
            # Factor 1: Mass Fragmentation (How small is the largest chunk?)
            # Range: [0, 1]. 0 means MaxFree == TotalFree (Good).
            # frag_mass = 1.0 - (max_free_size / free_size)

            # Factor 2: Hole Density (How porous is the memory layout?)
            # Range: [0, 1]. High means many holes relative to total blocks.
            # frag_holes = free_blocks / total_blocks

            # Composite Index
            # If Mass Frag is High, ratio is High.
            # If Mass Frag is Low (Good), we penalize it if Hole Density is High.
            # frag_ratio = frag_mass + (1.0 - frag_mass) * frag_holes

            frag_ratio = 1 - (max_free_size / free_size) * (
                used_blocks / total_blocks
            )
        else:
            # No free memory means 0 fragmentation (Fully utilized)
            frag_ratio = 0.0

        # C. Summary Row (Total -> Used -> Free)
        summary_rows.append(
            [
                allocator_name,
                total_blocks,
                used_blocks,
                free_blocks,
                format_size(total_size),
                format_size(used_size),
                format_size(free_size),
                format_size(max_used_size),
                format_size(max_free_size),
                f"{frag_ratio:.2%}",  # The new composite metric
            ]
        )

        # D. Distribution Rows
        dist_rows.append(
            [allocator_name, "Free Blocks"] + [b[0] for b in buckets]
        )
        dist_rows.append(
            [allocator_name, "Used Blocks"] + [b[1] for b in buckets]
        )

    # --- 3. Render Outputs ---
    sum_headers = [
        "ID",
        "Tot Blks",
        "Used Blks",
        "Free Blks",
        "Tot Size",
        "Used Size",
        "Free Size",
        "Max Used",
        "Max Free",
        "Frag Ratio*",
    ]
    print_table("Allocator Summary Statistics", sum_headers, summary_rows)
    print(
        " * Frag_Ratio = Frag_Mass + (1 - Frag_Mass) x (Free_Blks / Tot_Blks)"
    )
    print(" * Frag_Mass  = 1 - (Max_Free / Free_Size)")

    dist_headers = ["Allocator ID", "Block Type", *RANGE_HEADERS]
    print_table("Block Size Distribution", dist_headers, dist_rows)
