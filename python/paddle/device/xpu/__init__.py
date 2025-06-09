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
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from typing_extensions import TypeAlias

from paddle.base import core
from paddle.utils import deprecated

if TYPE_CHECKING:
    from paddle import XPUPlace

    _XPUPlaceLike: TypeAlias = Union[
        XPUPlace,
        str,  # some str like 'xpu:0', 'xpu:1', etc.
        int,  # some int like 0, 1, etc.
    ]

__all__ = [
    'synchronize',
    'device_count',
    'set_debug_level',
    'empty_cache',
    'max_memory_allocated',
    'max_memory_reserved',
    'reset_max_memory_allocated',
    'reset_max_memory_reserved',
    'memory_allocated',
    'memory_reserved',
    'memory_total',  # memory maneged by runtime, not paddle
    'memory_used',  # memory maneged by runtime, not paddle
]


def extract_xpu_device_id(device: _XPUPlaceLike, op_name: str) -> int:
    '''
    Return the id of the given xpu device. It is just a utility that will not be exposed to users.

    Args:
        device(paddle.XPUPlace or int or str): The device, the id of the device or
            the string name of device like 'xpu:x'.
            Default: None.

    Return:
        int: The id of the given device. If device is None, return the id of current device.
    '''
    if device is None:
        return core.get_xpu_current_device_id()

    if isinstance(device, int):
        device_id = device
    elif isinstance(device, core.XPUPlace):
        device_id = device.get_device_id()
    elif isinstance(device, str):
        if device.startswith('xpu:'):
            device_id = int(device[4:])
        else:
            raise ValueError(
                f"The current string {device} is not expected. Because {op_name} only support string which is like 'xpu:x'. "
                "Please input appropriate string again!"
            )
    else:
        raise ValueError(
            f"The device type {device} is not expected. Because {op_name} only support int, str or paddle.XPUPlace. "
            "Please input appropriate device again!"
        )

    assert (
        device_id >= 0
    ), f"The device id must be not less than 0, but got id = {device_id}."
    assert (
        device_id < device_count()
    ), f"The device id {device_id} exceeds xpu card number {device_count()}"
    return device_id


@deprecated(
    since="2.5.0",
    update_to="paddle.device.synchronize",
    level=1,
    reason="synchronize in paddle.device.xpu will be removed in future",
)
def synchronize(device: _XPUPlaceLike | None = None) -> int:
    """
    Wait for the compute on the given XPU device to finish.

    Parameters:
        device(paddle.XPUPlace()|int, optional): The device or the ID of the device.
        If device is None, the device is the current device. Default: None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:XPU)
            >>> import paddle
            >>> paddle.device.set_device('xpu')
            >>> paddle.device.xpu.synchronize()
            >>> paddle.device.xpu.synchronize(0)
            >>> paddle.device.xpu.synchronize(paddle.XPUPlace(0))

    """

    device_id = -1

    if device is not None:
        if isinstance(device, int):
            device_id = device
        elif isinstance(device, core.XPUPlace):
            device_id = device.get_device_id()
        else:
            raise ValueError("device type must be int or paddle.XPUPlace")

    return core._xpu_device_synchronize(device_id)


def device_count() -> int:
    '''
    Return the number of XPUs available.

    Returns:
        int: the number of XPUs available.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.device.xpu.device_count()

    '''

    num_xpus = (
        core.get_xpu_device_count()
        if hasattr(core, 'get_xpu_device_count')
        else 0
    )

    return num_xpus


def set_debug_level(level: int = 0) -> None:
    '''
    Set the debug level of XPUs' api. The default level is 0, which means no debug info.

    Args:
        int: Debug level of XPUs available.
            Level 0x1 for trace (Print the invocation of the interface),
            0x10 for checksum (Print the checksum of the tensor),
            0x100 for dump (Save the tensor as a file in npy format),
            0x1000 for profiling (Record the execution time of each operator).

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:XPU)
            >>> import paddle
            >>> paddle.device.set_device('xpu')

            >>> paddle.device.xpu.set_debug_level(1)
    '''
    name = "paddle.device.xpu.set_debug_level"
    if not core.is_compiled_with_xpu():
        raise ValueError(
            f"The API {name} is only supported in XPU PaddlePaddle. Please reinstall PaddlePaddle with XPU support to call this API."
        )
    else:
        core.set_xpu_debug_level(level)


def empty_cache() -> None:
    '''
    Releases idle cached memory held by the allocator so that those can be used in other XPU
    application and visible in `xpu-smi`. In most cases you don't need to use this function,
    Paddle does not release the memory back to the OS when you remove Tensors on the XPU,
    Because it keeps xpu memory in a pool so that next allocations can be done much faster.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:XPU)
            >>> import paddle
            >>> paddle.device.set_device('xpu')

            >>> tensor = paddle.randn([512, 512, 512], "float64")
            >>> del tensor
            >>> paddle.device.xpu.empty_cache()
    '''
    name = "paddle.device.xpu.empty_cache"
    if not core.is_compiled_with_xpu():
        raise ValueError(
            f"The API {name} is only supported in XPU PaddlePaddle. Please reinstall PaddlePaddle with XPU support to call this API."
        )
    else:
        core.xpu_empty_cache()


def max_memory_allocated(device: _XPUPlaceLike | None = None) -> int:
    '''
    Return the peak size of xpu memory that is allocated to tensor of the given device.

    Note:
        The size of XPU memory allocated to tensor is 256-byte aligned in Paddle, which may larger than the memory size that tensor actually need.
        For instance, a float32 0-D Tensor with shape [] in XPU will take up 256 bytes memory, even though storing a float32 data requires only 4 bytes.

    Args:
        device(paddle.XPUPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'xpu:x'. If device is None, the device is the current device.
            Default: None.

    Return:
        int: The peak size of xpu memory that is allocated to tensor of the given device, in bytes.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:XPU)
            >>> import paddle
            >>> paddle.device.set_device('xpu')

            >>> max_memory_allocated_size = paddle.device.xpu.max_memory_allocated(paddle.XPUPlace(0))
            >>> max_memory_allocated_size = paddle.device.xpu.max_memory_allocated(0)
            >>> max_memory_allocated_size = paddle.device.xpu.max_memory_allocated("xpu:0")
    '''
    name = "paddle.device.xpu.max_memory_allocated"
    if not core.is_compiled_with_xpu():
        raise ValueError(
            f"The API {name} is only supported in XPU PaddlePaddle. Please reinstall PaddlePaddle with XPU support to call this API."
        )
    device_id = extract_xpu_device_id(device, op_name=name)
    return core.device_memory_stat_peak_value("Allocated", device_id)


def max_memory_reserved(device: _XPUPlaceLike | None = None) -> int:
    '''
    Return the peak size of XPU memory that is held by the allocator of the given device.

    Args:
        device(paddle.XPUPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'xpu:x'. If device is None, the device is the current device.
            Default: None.

    Return:
        int: The peak size of XPU memory that is held by the allocator of the given device, in bytes.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:XPU)
            >>> import paddle
            >>> paddle.device.set_device('xpu')

            >>> max_memory_reserved_size = paddle.device.xpu.max_memory_reserved(paddle.XPUPlace(0))
            >>> max_memory_reserved_size = paddle.device.xpu.max_memory_reserved(0)
            >>> max_memory_reserved_size = paddle.device.xpu.max_memory_reserved("xpu:0")
    '''
    name = "paddle.device.xpu.max_memory_reserved"
    if not core.is_compiled_with_xpu():
        raise ValueError(
            f"The API {name} is only supported in XPU PaddlePaddle. Please reinstall PaddlePaddle with XPU support to call this API."
        )
    device_id = extract_xpu_device_id(device, op_name=name)
    return core.device_memory_stat_peak_value("Reserved", device_id)


def reset_max_memory_allocated(device: _XPUPlaceLike | None = None) -> None:
    '''
    Reset the peak size of XPU memory that is allocated to tensor of the given device.

    Args:
        device(paddle.XPUPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'xpu:x'. If device is None, the device is the current device.
            Default: None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:XPU)
            >>> import paddle
            >>> paddle.device.set_device('xpu')

            >>> paddle.device.xpu.reset_max_memory_allocated(paddle.XPUPlace(0))
            >>> paddle.device.xpu.reset_max_memory_allocated(0)
            >>> paddle.device.xpu.reset_max_memory_allocated("xpu:0")
    '''

    name = "paddle.device.xpu.reset_max_memory_allocated"
    if not core.is_compiled_with_xpu():
        raise ValueError(
            f"The API {name} is only supported in XPU PaddlePaddle. Please reinstall PaddlePaddle with XPU support to call this API."
        )
    device_id = extract_xpu_device_id(device, op_name=name)
    core.device_memory_stat_reset_peak_value("Allocated", device_id)


def reset_max_memory_reserved(device: _XPUPlaceLike | None = None) -> None:
    '''
    Reset the peak size of XPU memory that is held by the allocator of the given device.

    Args:
        device(paddle.XPUPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'xpu:x'. If device is None, the device is the current device.
            Default: None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:XPU)
            >>> import paddle
            >>> paddle.device.set_device('xpu')

            >>> paddle.device.xpu.reset_max_memory_reserved(paddle.XPUPlace(0))
            >>> paddle.device.xpu.reset_max_memory_reserved(0)
            >>> paddle.device.xpu.reset_max_memory_reserved("xpu:0")
    '''

    name = "paddle.device.xpu.reset_max_memory_reserved"
    if not core.is_compiled_with_xpu():
        raise ValueError(
            f"The API {name} is only supported in XPU PaddlePaddle. Please reinstall PaddlePaddle with XPU support to call this API."
        )
    device_id = extract_xpu_device_id(device, op_name=name)
    core.device_memory_stat_reset_peak_value("Reserved", device_id)


def memory_allocated(device: _XPUPlaceLike | None = None) -> int:
    '''
    Return the current size of xpu memory that is allocated to tensor of the given device.

    Note:
        The size of XPU memory allocated to tensor is 256-byte aligned in Paddle, which may be larger than the memory size that tensor actually need.
        For instance, a float32 0-D Tensor with shape [] in XPU will take up 256 bytes memory, even though storing a float32 data requires only 4 bytes.

    Args:
        device(paddle.XPUPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'xpu:x'. If device is None, the device is the current device.
            Default: None.

    Return:
        int: The current size of xpu memory that is allocated to tensor of the given device, in bytes.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:XPU)
            >>> import paddle
            >>> paddle.device.set_device('xpu')

            >>> memory_allocated_size = paddle.device.xpu.memory_allocated(paddle.XPUPlace(0))
            >>> memory_allocated_size = paddle.device.xpu.memory_allocated(0)
            >>> memory_allocated_size = paddle.device.xpu.memory_allocated("xpu:0")
    '''
    name = "paddle.device.xpu.memory_allocated"
    if not core.is_compiled_with_xpu():
        raise ValueError(
            f"The API {name} is only supported in XPU PaddlePaddle. Please reinstall PaddlePaddle with XPU support to call this API."
        )
    device_id = extract_xpu_device_id(device, op_name=name)
    return core.device_memory_stat_current_value("Allocated", device_id)


def memory_reserved(device: _XPUPlaceLike | None = None) -> int:
    '''
    Return the current size of XPU memory that is held by the allocator of the given device.

    Args:
        device(paddle.XPUPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'xpu:x'. If device is None, the device is the current device.
            Default: None.

    Return:
        int: The current size of XPU memory that is held by the allocator of the given device, in bytes.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:XPU)
            >>> import paddle
            >>> paddle.device.set_device('xpu')

            >>> memory_reserved_size = paddle.device.xpu.memory_reserved(paddle.XPUPlace(0))
            >>> memory_reserved_size = paddle.device.xpu.memory_reserved(0)
            >>> memory_reserved_size = paddle.device.xpu.memory_reserved("xpu:0")
    '''
    name = "paddle.device.xpu.memory_reserved"
    if not core.is_compiled_with_xpu():
        raise ValueError(
            f"The API {name} is only supported in XPU PaddlePaddle. Please reinstall PaddlePaddle with XPU support to call this API."
        )
    device_id = extract_xpu_device_id(device, op_name=name)
    return core.device_memory_stat_current_value("Reserved", device_id)


def memory_total(device: _XPUPlaceLike | None = None) -> int:
    '''
    Return the total size of XPU memory of the given device that is held by the XPU Runtime.

    Args:
        device(paddle.XPUPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'xpu:x'. If device is None, the device is the current device.
            Default: None.

    Return:
        int: The total size of XPU memory of the given device that is held by the XPU Runtime, in bytes.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:XPU)
            >>> import paddle
            >>> paddle.device.set_device('xpu')

            >>> memory_total_size = paddle.device.xpu.memory_total(paddle.XPUPlace(0))
            >>> memory_total_size = paddle.device.xpu.memory_total(0)
            >>> memory_total_size = paddle.device.xpu.memory_total("xpu:0")
    '''
    name = "paddle.device.xpu.memory_total"
    if not core.is_compiled_with_xpu():
        raise ValueError(
            f"The API {name} is only supported in XPU PaddlePaddle. Please reinstall PaddlePaddle with XPU support to call this API."
        )
    device_id = extract_xpu_device_id(device, op_name=name)
    return core.get_xpu_device_total_memory(device_id)


def memory_used(device: _XPUPlaceLike | None = None) -> int:
    '''
    Return the used size of XPU memory of the given device that is held by the XPU Runtime.

    Args:
        device(paddle.XPUPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'xpu:x'. If device is None, the device is the current device.
            Default: None.

    Return:
        int: The used size of XPU memory of the given device that is held by the XPU Runtime, in bytes.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:XPU)
            >>> import paddle
            >>> paddle.device.set_device('xpu')

            >>> memory_used_size = paddle.device.xpu.memory_used(paddle.XPUPlace(0))
            >>> memory_used_size = paddle.device.xpu.memory_used(0)
            >>> memory_used_size = paddle.device.xpu.memory_used("xpu:0")
    '''
    name = "paddle.device.xpu.memory_used"
    if not core.is_compiled_with_xpu():
        raise ValueError(
            f"The API {name} is only supported in XPU PaddlePaddle. Please reinstall PaddlePaddle with XPU support to call this API."
        )
    device_id = extract_xpu_device_id(device, op_name=name)
    return core.get_xpu_device_used_memory(device_id)
