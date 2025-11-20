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

from typing import TYPE_CHECKING, Union

from typing_extensions import TypeAlias

import paddle
from paddle.base import core

from .custom_streams import (  # noqa: F401
    Event,
    Stream,
    create_event,
    create_stream,
)

if TYPE_CHECKING:
    from paddle import CustomPlace

    _CustomPlaceLike: TypeAlias = Union[
        CustomPlace,
        str,  # some string like "iluvatar_gpu" "metax_gpu:0", etc.
        int,  # some int like 0, 1, etc.
    ]

dev_types = core.get_all_custom_device_type()

dev_type = dev_types[0] if dev_types else None

if dev_type and not core.is_compiled_with_custom_device(dev_type):
    raise Exception(
        "No custom device available, please install paddle with custom device support"
    )
if dev_type and dev_type in ['metax_gpu', 'iluvatar_gpu']:
    from .gpgpu_backend import get_device_properties
else:
    from .default_backend import get_device_properties

__all__ = [
    'Stream',
    'Event',
    'device_count',
    'get_device_properties',
    'empty_cache',
    'max_memory_allocated',
    'max_memory_reserved',
    'reset_max_memory_allocated',
    'reset_max_memory_reserved',
    'memory_allocated',
    'memory_reserved',
    'current_stream',
    'synchronize',
]


def device_count(device_type: str | None = None) -> int:
    '''
    Return the number of custom devices available.

    Args:
        device_type (str, optional): The type of custom device (e.g., 'npu', 'mlu', etc.).
            If None, returns the count of the first available custom device type.

    Returns:
        int: the number of custom devices available.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.device_count()
            >>> paddle.device.device_count('npu')
    '''

    if device_type:
        num = core.get_custom_device_count(device_type)
    else:
        num = core.get_custom_device_count(dev_type)

    return num


def empty_cache() -> None:
    '''
    Releases idle cached memory held by the allocator so that those can be used in other GPU
    application and visible in device-specific tools.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.empty_cache()
    '''
    core.device_empty_cache()


def max_memory_allocated(device: _CustomPlaceLike | None = None) -> int:
    '''
    Return the peak size of memory that is allocated to tensor of the given device.

    Args:
        device(_CustomPlaceLike, optional): Support input like 'npu:0', 'mlu', int, or CustomPlace.
            If None, the device is the first available custom device with index 0.

    Returns:
        int: The peak size of memory that is allocated to tensor of the given device, in bytes.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.max_memory_allocated('npu:0')
            >>> paddle.device.max_memory_allocated('npu')
            >>> paddle.device.max_memory_allocated(0)
            >>> paddle.device.max_memory_allocated(Paddle.CustomPlace('npu',0))
    '''
    device_id = 0

    if device is None:
        device_id = 0
    elif isinstance(device, str):
        colon_idx = device.rfind(':')
        if colon_idx == -1:
            device_id = 0
        else:
            device_id_str = device[colon_idx + 1 :]
            if not device_id_str.isdigit():
                raise ValueError(
                    f"Invalid device ID '{device_id_str}'. "
                    f"After colon must be digits only. "
                    "Example: 'npu:0'"
                )
            device_id = int(device_id_str)
    elif isinstance(device, int):
        device_id = device
    elif isinstance(device, core.CustomPlace):
        device_id = device.get_device_id()
    else:
        raise ValueError(
            f"The input: {device} is not expected. Because paddle.device."
            "max_memory_allocated only support str, int or CustomPlace. "
            "Please input appropriate device again! "
            "Example: 'npu:0'"
        )

    return core.device_memory_stat_peak_value("Allocated", device_id)


def max_memory_reserved(device: _CustomPlaceLike | None = None) -> int:
    '''
    Return the peak size of memory that is held by the allocator of the given device.

    Args:
        device(_CustomPlaceLike, optional): Support input like 'npu:0', 'mlu', int, or CustomPlace.
            If None, the device is the first available custom device with index 0.

    Returns:
        int: The peak size of memory that is held by the allocator of the given device, in bytes.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.max_memory_reserved('npu:0')
            >>> paddle.device.max_memory_reserved('npu')
            >>> paddle.device.max_memory_reserved(0)
            >>> paddle.device.max_memory_reserved(Paddle.CustomPlace('npu',0))
    '''
    device_id = 0

    if device is None:
        device_id = 0
    elif isinstance(device, str):
        colon_idx = device.rfind(':')
        if colon_idx == -1:
            device_id = 0
        else:
            device_id_str = device[colon_idx + 1 :]
            if not device_id_str.isdigit():
                raise ValueError(
                    f"Invalid device ID '{device_id_str}'. "
                    f"After colon must be digits only. "
                    "Example: 'npu:0'"
                )
            device_id = int(device_id_str)
    elif isinstance(device, int):
        device_id = device
    elif isinstance(device, core.CustomPlace):
        device_id = device.get_device_id()
    else:
        raise ValueError(
            f"The input: {device} is not expected. Because paddle.device."
            "max_memory_reserved only support str, int or CustomPlace. "
            "Please input appropriate device again! "
            "Example: 'npu:0'"
        )

    return core.device_memory_stat_peak_value("Reserved", device_id)


def reset_max_memory_allocated(device: _CustomPlaceLike | None = None) -> None:
    '''
    Reset the peak size of memory that is allocated to tensor of the given device.

    Args:
        device(_CustomPlaceLike, optional): Support input like 'npu:0', 'mlu', int, or CustomPlace.
            If None, the device is the first available custom device with index 0.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.reset_max_memory_allocated('npu:0')
            >>> paddle.device.reset_max_memory_allocated('npu')
            >>> paddle.device.reset_max_memory_allocated(0)
            >>> paddle.device.reset_max_memory_allocated(Paddle.CustomPlace('npu',0))
    '''
    device_id = 0

    if device is None:
        device_id = 0
    elif isinstance(device, str):
        colon_idx = device.rfind(':')
        if colon_idx == -1:
            device_id = 0
        else:
            device_id_str = device[colon_idx + 1 :]
            if not device_id_str.isdigit():
                raise ValueError(
                    f"Invalid device ID '{device_id_str}'. "
                    f"After colon must be digits only. "
                    "Example: 'npu:0'"
                )
            device_id = int(device_id_str)
    elif isinstance(device, int):
        device_id = device
    elif isinstance(device, core.CustomPlace):
        device_id = device.get_device_id()
    else:
        raise ValueError(
            f"The input: {device} is not expected. Because paddle.device."
            "reset_max_memory_allocated only support str, int or CustomPlace. "
            "Please input appropriate device again! "
            "Example: 'npu:0'"
        )

    core.device_memory_stat_reset_peak_value("Allocated", device_id)


def reset_max_memory_reserved(device: _CustomPlaceLike | None = None) -> None:
    '''
    Reset the peak size of memory that is held by the allocator of the given device.

    Args:
        device(_CustomPlaceLike, optional): Support input like 'npu:0', 'mlu', int, or CustomPlace.
            If None, the device is the first available custom device with index 0.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.reset_max_memory_reserved('npu:0')
            >>> paddle.device.reset_max_memory_reserved('npu')
            >>> paddle.device.reset_max_memory_reserved(0)
            >>> paddle.device.reset_max_memory_reserved(Paddle.CustomPlace('npu',0))
    '''
    device_id = 0

    if device is None:
        device_id = 0
    elif isinstance(device, str):
        colon_idx = device.rfind(':')
        if colon_idx == -1:
            device_id = 0
        else:
            device_id_str = device[colon_idx + 1 :]
            if not device_id_str.isdigit():
                raise ValueError(
                    f"Invalid device ID '{device_id_str}'. "
                    f"After colon must be digits only. "
                    "Example: 'npu:0'"
                )
            device_id = int(device_id_str)
    elif isinstance(device, int):
        device_id = device
    elif isinstance(device, core.CustomPlace):
        device_id = device.get_device_id()
    else:
        raise ValueError(
            f"The input: {device} is not expected. Because paddle.device."
            "reset_max_memory_reserved only support str, int or CustomPlace. "
            "Please input appropriate device again! "
            "Example: 'npu:0'"
        )

    core.device_memory_stat_reset_peak_value("Reserved", device_id)


def memory_allocated(device: _CustomPlaceLike | None = None) -> int:
    '''
    Return the current size of memory that is allocated to tensor of the given device.

    Args:
        device(_CustomPlaceLike, optional): Support input like 'npu:0', 'mlu', int, or CustomPlace.
            If None, the device is the first available custom device with index 0.

    Returns:
        int: The current size of memory that is allocated to tensor of the given device, in bytes.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.memory_allocated('npu:0')
            >>> paddle.device.memory_allocated('npu')
            >>> paddle.device.memory_allocated(0)
            >>> paddle.device.memory_allocated(Paddle.CustomPlace('npu',0))
    '''
    device_id = 0

    if device is None:
        device_id = 0
    elif isinstance(device, str):
        colon_idx = device.rfind(':')
        if colon_idx == -1:
            device_id = 0
        else:
            device_id_str = device[colon_idx + 1 :]
            if not device_id_str.isdigit():
                raise ValueError(
                    f"Invalid device ID '{device_id_str}'. "
                    f"After colon must be digits only. "
                    "Example: 'npu:0'"
                )
            device_id = int(device_id_str)
    elif isinstance(device, int):
        device_id = device
    elif isinstance(device, core.CustomPlace):
        device_id = device.get_device_id()
    else:
        raise ValueError(
            f"The input: {device} is not expected. Because paddle.device."
            "memory_allocated only support str, int or CustomPlace. "
            "Please input appropriate device again! "
            "Example: 'npu:0'"
        )

    return core.device_memory_stat_current_value("Allocated", device_id)


def memory_reserved(device: _CustomPlaceLike | None = None) -> int:
    '''
    Return the current size of memory that is held by the allocator of the given device.

    Args:
        device(_CustomPlaceLike, optional): Support input like 'npu:0', 'mlu', int, or CustomPlace.
            If None, the device is the first available custom device with index 0.

    Returns:
        int: The current size of memory that is held by the allocator of the given device, in bytes.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.memory_reserved('npu:0')
            >>> paddle.device.memory_reserved('npu')
            >>> paddle.device.memory_reserved(0)
            >>> paddle.device.memory_reserved(Paddle.CustomPlace('npu',0))
    '''
    device_id = 0

    if device is None:
        device_id = 0
    elif isinstance(device, str):
        colon_idx = device.rfind(':')
        if colon_idx == -1:
            device_id = 0
        else:
            device_id_str = device[colon_idx + 1 :]
            if not device_id_str.isdigit():
                raise ValueError(
                    f"Invalid device ID '{device_id_str}'. "
                    f"After colon must be digits only. "
                    "Example: 'npu:0'"
                )
            device_id = int(device_id_str)
    elif isinstance(device, int):
        device_id = device
    elif isinstance(device, core.CustomPlace):
        device_id = device.get_device_id()
    else:
        raise ValueError(
            f"The input: {device} is not expected. Because paddle.device."
            "memory_reserved only support str, int or CustomPlace. "
            "Please input appropriate device again! "
            "Example: 'npu:0'"
        )

    return core.device_memory_stat_current_value("Reserved", device_id)


def current_stream(device: _CustomPlaceLike | None = None) -> core.CustomStream:
    '''
    Return the current stream by the device.

    Args:
        device(_CustomPlaceLike, optional): Support input like 'npu:0', 'mlu', int, or CustomPlace.
            If None, the device is the first available custom device with index 0.

    Returns:
        Stream: The stream to the device.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.current_stream('npu:0')
            >>> paddle.device.current_stream('npu')
            >>> paddle.device.current_stream(0)
            >>> paddle.device.current_stream(Paddle.CustomPlace('npu',0))
    '''
    device_id = 0

    if device is None:
        device_id = 0
    elif isinstance(device, str):
        colon_idx = device.rfind(':')
        if colon_idx == -1:
            device_id = 0
        else:
            device_id_str = device[colon_idx + 1 :]
            if not device_id_str.isdigit():
                raise ValueError(
                    f"Invalid device ID '{device_id_str}'. "
                    f"After colon must be digits only. "
                    "Example: 'npu:0'"
                )
            device_id = int(device_id_str)
    elif isinstance(device, int):
        device_id = device
    elif isinstance(device, core.CustomPlace):
        device_id = device.get_device_id()
    else:
        raise ValueError(
            f"The input: {device} is not expected. Because paddle.device."
            "current_stream only support str, int or CustomPlace. "
            "Please input appropriate device again! "
            "Example: 'npu:0'"
        )

    return core._get_current_custom_device_stream(dev_type, device_id)


def synchronize(device: _CustomPlaceLike | None = None) -> None:
    """
    Wait for the compute on the given device to finish.

    Args:
        device(_CustomPlaceLike, optional): Support input like 'npu:0', 'mlu', int, or CustomPlace.
            If None, the device is the first available custom device with index 0.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.synchronize('npu:0')
            >>> paddle.device.synchronize('npu')
            >>> paddle.device.synchronize(0)
            >>> paddle.device.synchronize(Paddle.CustomPlace('npu',0))
    """
    device_id = 0

    if device is None:
        device_id = 0
    elif isinstance(device, str):
        colon_idx = device.rfind(':')
        if colon_idx == -1:
            device_id = 0
        else:
            device_id_str = device[colon_idx + 1 :]
            if not device_id_str.isdigit():
                raise ValueError(
                    f"Invalid device ID '{device_id_str}'. "
                    f"After colon must be digits only. "
                    "Example: 'npu:0'"
                )
            device_id = int(device_id_str)
    elif isinstance(device, int):
        device_id = device
    elif isinstance(device, core.CustomPlace):
        device_id = device.get_device_id()
    else:
        raise ValueError(
            f"The input: {device} is not expected. Because paddle.device."
            "synchronize only support str, int or CustomPlace. "
            "Please input appropriate device again! "
            "Example: 'npu:0'"
        )

    core._synchronize_custom_device(dev_type, device_id)


def get_rng_state(
    device: _CustomPlaceLike | None = None,
) -> core.GeneratorState:
    r'''
    Get the random state for the default generator.

    Returns:
        Tensor: The random state tensor.

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle
            >>> paddle.device.get_rng_state()

    '''
    place = paddle.device.device_to_place(device)
    if isinstance(place, core.CPUPlace):
        return core.default_cpu_generator().get_state()
    return core.default_custom_device_generator(place).get_state()


def set_rng_state(
    new_state: core.GeneratorState, device: _CustomPlaceLike | None = None
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
        core.default_custom_device_generator(place).set_state(new_state)


def manual_seed(seed: int) -> None:
    r"""Set the seed for generating random numbers for the current Device.

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

    """
    seed = int(seed)
    place = paddle.framework._current_expected_place()
    if isinstance(place, core.CPUPlace):
        core.default_cpu_generator().manual_seed(seed)
    else:
        core.default_custom_device_generator(place).manual_seed(seed)
