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

import ctypes
import os
import re
from typing import TYPE_CHECKING, Union

from typing_extensions import TypeAlias

import paddle
from paddle.base import core, framework
from paddle.base.framework import (
    is_compiled_with_cinn,
    is_compiled_with_cuda,
    is_compiled_with_distribute,
    is_compiled_with_rocm,
)

from . import (  # noqa: F401
    cuda,
    xpu,
)

if TYPE_CHECKING:
    from types import TracebackType

    from paddle import IPUPlace as _IPUPlace, XPUPlace as _XPUPlace
    from paddle._typing.device_like import PlaceLike
    from paddle.base.core import Place

    _InitStreamBase = Union[core.CUDAStream, core.CustomDeviceStream]
    _InitEventBase = Union[core.CUDAEvent, core.CustomDeviceEvent]

    from paddle import CUDAPlace, CustomPlace
    from paddle.base.libpaddle import _customDeviceProperties

    _CustomPlaceLike: TypeAlias = Union[
        CUDAPlace,
        CustomPlace,
        str,  # some string like "iluvatar_gpu" "metax_gpu:0", etc.
        int,  # some int like 0, 1, etc.
    ]

__all__ = [
    'get_cudnn_version',
    'set_device',
    'get_device',
    'XPUPlace',
    'IPUPlace',
    'is_compiled_with_xpu',
    'is_compiled_with_ipu',
    'is_compiled_with_cinn',
    'is_compiled_with_cuda',
    'is_compiled_with_rocm',
    'is_compiled_with_distribute',
    'is_compiled_with_custom_device',
    'get_all_device_type',
    'get_all_custom_device_type',
    'get_available_device',
    'get_available_custom_device',
    'get_device_properties',
    'Stream',
    'Event',
    'current_stream',
    'set_stream',
    'stream_guard',
    'device_guard',
    'synchronize',
]

_cudnn_version = None


def is_compiled_with_custom_device(device_type: str) -> bool:
    """

    Whether paddle was built with Paddle_CUSTOM_DEVICE .

    Args:
        device_type (str): the registered device type, like "npu".

    Return:
        bool, ``True`` if CustomDevice is supported, otherwise ``False``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> support_npu = paddle.device.is_compiled_with_custom_device("npu")

    """
    return core.is_compiled_with_custom_device(device_type)


def is_compiled_with_ipu() -> bool:
    """

    Whether paddle was built with WITH_IPU=ON to support Graphcore IPU.

    Returns (bool): `True` if IPU is supported, otherwise `False`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> support_ipu = paddle.is_compiled_with_ipu()

    """
    return core.is_compiled_with_ipu()


def IPUPlace() -> _IPUPlace:
    """

    Return a Graphcore IPU Place

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:IPU)

            >>> import paddle
            >>> paddle.device.set_device('ipu')
            >>> place = paddle.device.IPUPlace()

    """
    return core.IPUPlace()


def is_compiled_with_xpu() -> bool:
    """

    Whether paddle was built with WITH_XPU=ON to support Baidu Kunlun

    Returns (bool): whether paddle was built with WITH_XPU=ON

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> support_xpu = paddle.device.is_compiled_with_xpu()

    """
    return core.is_compiled_with_xpu()


def XPUPlace(dev_id: int) -> _XPUPlace:
    """

    Return a Baidu Kunlun Place

    Args:
        dev_id(int): Baidu Kunlun device id

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:XPU)

            >>> import paddle
            >>> paddle.device.set_device('xpu')
            >>> place = paddle.device.XPUPlace(0)

    """
    return core.XPUPlace(dev_id)


def get_cudnn_version() -> int | None:
    """

    This function return the version of cudnn. the return value is int which represents the
    cudnn version. For example, if it return 7600, it represents the version of cudnn is 7.6.

    Returns:
        int: A int value which represents the cudnn version. If cudnn version is not installed, it return None.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> cudnn_version = paddle.device.get_cudnn_version()



    """
    global _cudnn_version
    if not core.is_compiled_with_cuda():
        return None
    if _cudnn_version is None:
        cudnn_version = int(core.cudnn_version())
        _cudnn_version = cudnn_version
        if _cudnn_version < 0:
            return None
        else:
            return cudnn_version
    else:
        return _cudnn_version


def _convert_to_place(device: str) -> PlaceLike:
    lower_device = device.lower()
    if device in core.get_all_custom_device_type():
        selected_devices = os.getenv(f"FLAGS_selected_{device}s", "0").split(
            ","
        )
        device_id = int(selected_devices[0])
        place = core.CustomPlace(device, device_id)
    elif lower_device == 'cpu':
        place = core.CPUPlace()
    elif lower_device == 'gpu' or lower_device == 'dcu':
        if not core.is_compiled_with_cuda():
            raise ValueError(
                "The device should not be 'gpu', "
                "since PaddlePaddle is not compiled with CUDA"
            )
        place = core.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
    elif lower_device == 'xpu':
        if not core.is_compiled_with_xpu():
            raise ValueError(
                "The device should not be 'xpu', "
                "since PaddlePaddle is not compiled with XPU"
            )
        selected_xpus = os.getenv("FLAGS_selected_xpus", "0").split(",")
        device_id = int(selected_xpus[0])
        place = core.XPUPlace(device_id)
    elif lower_device == 'ipu':
        if not core.is_compiled_with_ipu():
            raise ValueError(
                "The device should not be 'ipu', "
                "since PaddlePaddle is not compiled with IPU"
            )
        place = core.IPUPlace()
    else:
        available_gpu_device = re.match(r'gpu:\d+', lower_device) or re.match(
            r'dcu:\d+', lower_device
        )
        available_xpu_device = re.match(r'xpu:\d+', lower_device)
        if available_gpu_device:
            if not core.is_compiled_with_cuda():
                raise ValueError(
                    f"The device should not be {available_gpu_device}, since PaddlePaddle is "
                    "not compiled with CUDA"
                )
            device_info_list = device.split(':', 1)
            device_id = device_info_list[1]
            device_id = int(device_id)
            place = core.CUDAPlace(device_id)
        if available_xpu_device:
            if not core.is_compiled_with_xpu():
                raise ValueError(
                    f"The device should not be {available_xpu_device}, since PaddlePaddle is "
                    "not compiled with XPU"
                )
            device_info_list = device.split(':', 1)
            device_id = device_info_list[1]
            device_id = int(device_id)
            place = core.XPUPlace(device_id)
        if not available_gpu_device and not available_xpu_device:
            device_info_list = device.split(':', 1)
            device_type = device_info_list[0]
            if device_type in core.get_all_custom_device_type():
                device_id = device_info_list[1]
                device_id = int(device_id)
                place = core.CustomPlace(device_type, device_id)
            else:
                raise ValueError(
                    "The device must be a string which is like 'cpu', {}".format(
                        ', '.join(
                            f"'{x}', '{x}:x'"
                            for x in [
                                'gpu',
                                'dcu',
                                'xpu',
                                'npu',
                                *core.get_all_custom_device_type(),
                            ]
                        )
                    )
                )
    return place


def set_device(device: str) -> PlaceLike:
    """

    Paddle supports running calculations on various types of devices, including CPU, GPU, XPU, NPU and IPU.
    They are represented by string identifiers. This function can specify the global device
    which the OP will run.

    Args:
        device(str): This parameter determines the specific running device.
            It can be ``cpu``, ``gpu``, ``xpu``, ``npu``, ``gpu:x``, ``xpu:x``, ``npu:x`` and ``ipu``,
            where ``x`` is the index of the GPUs, XPUs or NPUs.

    Returns:
        Place,the Place to set.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> paddle.device.set_device("cpu")
            >>> x1 = paddle.ones(name='x1', shape=[1, 2], dtype='int32')
            >>> x2 = paddle.zeros(name='x2', shape=[1, 2], dtype='int32')
            >>> data = paddle.stack([x1,x2], axis=1)

    """
    place = _convert_to_place(device)
    framework._set_expected_place(place)
    return place


def get_device() -> str:
    """

    This function can get the current global device of the program is running.
    It's a string which is like 'cpu', 'gpu:x', 'xpu:x' and 'npu:x'. if the global device is not
    set, it will return a string which is 'gpu:x' when cuda is available or it
    will return a string which is 'cpu' when cuda is not available.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> device = paddle.device.get_device()

    """
    device = ''
    place = framework._current_expected_place_()
    if isinstance(place, core.CPUPlace):
        device = 'cpu'
    elif isinstance(place, core.CUDAPlace):
        device_id = place.get_device_id()
        device = 'gpu:' + str(device_id)
    elif isinstance(place, core.XPUPlace):
        device_id = place.get_device_id()
        device = 'xpu:' + str(device_id)
    elif isinstance(place, core.IPUPlace):
        num_devices = core.get_ipu_device_count()
        device = f"ipus:{{0-{num_devices - 1}}}"
    elif isinstance(place, core.CustomPlace):
        device_id = place.get_device_id()
        device_type = place.get_device_type()
        device = device_type + ':' + str(device_id)
    else:
        raise ValueError(f"The device specification {place} is invalid")

    return device


def device_count(dev_type: str | None = None) -> int:
    '''
    Return the number of devices available.
    Args:
        dev_type (str, optional): Device type string, e.g., 'gpu', 'npu', etc.
        If None, will return the number of CUDA devices if available,
        otherwise the first available custom device count.
    Returns:
        int: the number of devices available.
    Examples:
        .. code-block:: python
            >>> import paddle
            >>> paddle.device.device_count()
            >>> paddle.device.device_count('gpu')
            >>> paddle.device.device_count('npu')
    '''
    if dev_type is None:
        if paddle.is_compiled_with_cuda():
            num = (
                core.get_cuda_device_count()
                if hasattr(core, 'get_cuda_device_count')
                else 0
            )
        elif hasattr(core, 'get_all_custom_device_type'):
            custom_types = core.get_all_custom_device_type()
            if custom_types:
                num = (
                    core.get_custom_device_count(custom_types[0])
                    if hasattr(core, 'get_custom_device_count')
                    else 0
                )
            else:
                num = 0
        else:
            raise ValueError(
                "Paddle is not compiled with GPU or Custom Device."
            )
        return num

    if dev_type == 'gpu':
        if paddle.is_compiled_with_cuda():
            num = (
                core.get_cuda_device_count()
                if hasattr(core, 'get_cuda_device_count')
                else 0
            )
        else:
            raise ValueError("Paddle is not compiled with GPU.")
    else:
        if hasattr(
            core, 'is_compiled_with_custom_device'
        ) and core.is_compiled_with_custom_device(dev_type):
            num = (
                core.get_custom_device_count(dev_type)
                if hasattr(core, 'get_custom_device_count')
                else 0
            )
        else:
            raise ValueError(
                f"Unsupported or unavailable device type: {dev_type}"
            )
    return num


def get_all_device_type() -> list[str]:
    """

    Get all available device types.

    Returns:
        A list of all available device types.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.get_all_device_type()

            >>> # Case 1: paddlepaddle-cpu package installed, and no custom device registered.
            >>> # Output: ['cpu']

            >>> # Case 2: paddlepaddle-gpu package installed, and no custom device registered.
            >>> # Output: ['cpu', 'gpu']

            >>> # Case 3: paddlepaddle-cpu package installed, and custom device 'CustomCPU' is registered.
            >>> # Output: ['cpu', 'CustomCPU']

            >>> # Case 4: paddlepaddle-gpu package installed, and custom device 'CustomCPU' and 'CustomGPU' is registered.
            >>> # Output: ['cpu', 'gpu', 'CustomCPU', 'CustomGPU']

    """
    return core.get_all_device_type()


def get_all_custom_device_type() -> list[str] | None:
    """

    Get all available custom device types.

    Returns:
        A list of all available custom device types.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.get_all_custom_device_type()

            >>> # Case 1: paddlepaddle-gpu package installed, and no custom device registered.
            >>> # Output: None

            >>> # Case 2: paddlepaddle-gpu package installed, and custom device 'CustomCPU' and 'CustomGPU' is registered.
            >>> # Output: ['CustomCPU', 'CustomGPU']

    """
    return core.get_all_custom_device_type()


def get_available_device() -> list[str]:
    """

    Get all available devices.

    Returns:
        A list of all available devices.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.get_available_device()

            >>> # Case 1: paddlepaddle-cpu package installed, and no custom device registered.
            >>> # Output: ['cpu']

            >>> # Case 2: paddlepaddle-gpu package installed, and no custom device registered.
            >>> # Output: ['cpu', 'gpu:0', 'gpu:1']

            >>> # Case 3: paddlepaddle-cpu package installed, and custom device 'CustomCPU' is registered.
            >>> # Output: ['cpu', 'CustomCPU']

            >>> # Case 4: paddlepaddle-gpu package installed, and custom device 'CustomCPU' and 'CustomGPU' is registered.
            >>> # Output: ['cpu', 'gpu:0', 'gpu:1', 'CustomCPU', 'CustomGPU:0', 'CustomGPU:1']

    """
    return core.get_available_device()


def get_available_custom_device() -> list[str] | None:
    """

    Get all available custom devices.

    Returns:
       A list of all available custom devices.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.get_available_custom_device()

            >>> # Case 1: paddlepaddle-gpu package installed, and no custom device registered.
            >>> # Output: None

            >>> # Case 2: paddlepaddle-gpu package installed, and custom device 'CustomCPU' and 'CustomGPU' is registered.
            >>> # Output: ['CustomCPU', 'CustomGPU:0', 'CustomGPU:1']

    """
    return core.get_available_custom_device()


def get_device_properties(
    device: _CustomPlaceLike | None = None,
) -> _customDeviceProperties:
    """

    Return the properties of given device.

    Args:
        device(|paddle.CustomPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like npu:x' which to get the properties of the
            device from. If device is None, the device is the current device.
            Default: None.

    Returns:
       _customDeviceProperties: The properties of the device which include ASCII string
        identifying device, major compute capability, minor compute capability, global
        memory available and the number of multiprocessors on the device.

    Examples:
        .. code-block:: python

            >>> # import paddle
            >>> # paddle.device.set_device('npu')
            >>> # paddle.device.get_device_properties('npu:0')
            >>> # _customDeviceProperties(name='', major=0, minor=0, total_memory=0MB, multi_processor_count=0)

            >>> # paddle.device.get_device_properties('npu')
            >>> # _customDeviceProperties(name='', major=0, minor=0, total_memory=0MB, multi_processor_count=0)
    """
    device_name = None

    if device is not None:
        if isinstance(device, str):
            colon_idx = device.rfind(':')
            if colon_idx == -1:
                device_name = device
                device_id = 0
            else:
                device_name = device[:colon_idx]
                device_id_str = device[colon_idx + 1 :]

                if not device_id_str.isdigit():
                    raise ValueError(
                        f"Invalid device ID '{device_id_str}'. "
                        f"After colon must be digits only. "
                        "Example: 'metax_gpu:0'"
                    )

                device_id = int(device_id_str)

        else:
            raise ValueError(
                f"The input: {device} is not expected. Because paddle.device."
                "get_device_properties only support str. "
                "Please input appropriate device again!"
                "Example: 'metax_gpu:0'"
            )
    else:
        raise ValueError(
            f"The input: {device} is not expected. Because paddle.device."
            "get_device_properties only support str. "
            "Please input appropriate device again!"
            "Example: 'metax_gpu:0'"
        )
    if not core.is_compiled_with_custom_device(device_name):
        raise ValueError(
            f"PaddlePaddle is not compiled with support for '{device_name}' device. "
            "Please reinstall PaddlePaddle with Custom Device support "
            "to call this API."
        )

    return core.get_device_properties(device_name, device_id)


def extract_device_id(device: _CustomPlaceLike, op_name: str) -> int:
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

    assert (
        device_id >= 0
    ), f"The device id must be not less than 0, but got id = {device_id}."

    if core.is_compiled_with_cuda():
        assert (
            device_id < device_count()
        ), f"The device id {device_id} exceeds gpu card number {device_count()}"
    else:
        assert device_id < core.get_custom_device_count(
            device_type
        ), f"The device id {device_id} exceeds {device_type} device card number {core.get_custom_device_count(device_type)}"
    return device_id


def max_memory_allocated(device: _CustomPlaceLike | None = None) -> int:
    '''
    Return the peak size of memory that is allocated to tensor of the given device. This

    Note:
        The size of memory allocated to tensor is 256-byte aligned in Paddle, which may larger than the memory size that tensor actually need.
        For instance, a float32 0-D Tensor with shape [] will take up 256 bytes memory, even though storing a float32 data requires only 4 bytes.

    Args:
        device(paddle.CUDAPlace|paddle.CustomPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    Return:
        int: The peak size of memory that is allocated to tensor of the given device, in bytes.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> max_memory_allocated_size = paddle.device.max_memory_allocated(paddle.CUDAPlace(0))
            >>> max_memory_allocated_size = paddle.device.max_memory_allocated(0)
            >>> max_memory_allocated_size = paddle.device.max_memory_allocated("gpu:0")
    '''
    name = "paddle.device.max_memory_allocated"
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
    device_id = extract_device_id(device, op_name=name)
    return core.device_memory_stat_peak_value("Allocated", device_id)


def max_memory_reserved(device: _CustomPlaceLike | None = None) -> int:
    '''
    Return the peak size of memory that is held by the allocator of the given device.

    Args:
        device(paddle.CUDAPlace|paddle.CustomPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    Return:
        int: The peak size of memory that is held by the allocator of the given device, in bytes.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> max_memory_reserved_size = paddle.device.max_memory_reserved(paddle.CUDAPlace(0))
            >>> max_memory_reserved_size = paddle.device.max_memory_reserved(0)
            >>> max_memory_reserved_size = paddle.device.max_memory_reserved("gpu:0")
    '''
    name = "paddle.device.max_memory_reserved"
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
    device_id = extract_device_id(device, op_name=name)
    return core.device_memory_stat_peak_value("Reserved", device_id)


def reset_max_memory_allocated(device: _CustomPlaceLike | None = None) -> None:
    '''
    Reset the peak size of memory that is allocated to tensor of the given device.

    Args:
        device(paddle.CUDAPlace|paddle.CustomPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> paddle.device.reset_max_memory_allocated(paddle.CUDAPlace(0))
            >>> paddle.device.reset_max_memory_allocated(0)
            >>> paddle.device.reset_max_memory_allocated("gpu:0")
    '''

    name = "paddle.device.reset_max_memory_allocated"
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
    device_id = extract_device_id(device, op_name=name)
    core.device_memory_stat_reset_peak_value("Allocated", device_id)


def reset_max_memory_reserved(device: _CustomPlaceLike | None = None) -> None:
    '''
    Reset the peak size of memory that is held by the allocator of the given device.

    Args:
        device(paddle.CUDAPlace|paddle.CustomPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> paddle.device.reset_max_memory_reserved(paddle.CUDAPlace(0))
            >>> paddle.device.reset_max_memory_reserved(0)
            >>> paddle.device.reset_max_memory_reserved("gpu:0")
    '''

    name = "paddle.device.reset_max_memory_reserved"
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
    device_id = extract_device_id(device, op_name=name)
    core.device_memory_stat_reset_peak_value("Reserved", device_id)


def memory_allocated(device: _CustomPlaceLike | None = None) -> int:
    '''
    Return the current size of memory that is allocated to tensor of the given device.

    Note:
        The size of memory allocated to tensor is 256-byte aligned in Paddle, which may be larger than the memory size that tensor actually need.
        For instance, a float32 0-D Tensor with shape [] will take up 256 bytes memory, even though storing a float32 data requires only 4 bytes.

    Args:
        device(paddle.CUDAPlace|paddle.CustomPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like 'gpu:x'. If device is None, the device is the current device.
            Default: None.

    Return:
        int: The current size of memory that is allocated to tensor of the given device, in bytes.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> memory_allocated_size = paddle.device.memory_allocated(paddle.CUDAPlace(0))
            >>> memory_allocated_size = paddle.device.memory_allocated(0)
            >>> memory_allocated_size = paddle.device.memory_allocated("gpu:0")
    '''
    name = "paddle.device.memory_allocated"
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
    device_id = extract_device_id(device, op_name=name)
    return core.device_memory_stat_current_value("Allocated", device_id)


def memory_reserved(device: _CustomPlaceLike | None = None) -> int:
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

            >>> memory_reserved_size = paddle.device.memory_reserved(paddle.CUDAPlace(0))
            >>> memory_reserved_size = paddle.device.memory_reserved(0)
            >>> memory_reserved_size = paddle.device.memory_reserved("gpu:0")
    '''
    name = "paddle.device.memory_reserved"
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
    device_id = extract_device_id(device, op_name=name)
    return core.device_memory_stat_current_value("Reserved", device_id)


class Event:
    '''

    A device event wrapper around StreamBase.

    Args:
        device(str|paddle.CUDAPlace(n)|paddle.CustomPlace(n)|None): Which device the stream run on. If device is None, the device is the current device. Default: None.
            It can be ``gpu``, ``gpu:x``, ``custom_device``, ``custom_device:x``, where ``custom_device`` is the name of CustomDevice,
            where ``x`` is the index of the GPUs, XPUs. And it can be paddle.CUDAPlace(n) or paddle.CustomPlace(n).
        enable_timing (bool, optional): indicates if the event should measure time, default is False
        blocking (bool, optional): if True, ``wait`` will be blocking, default is False
        interprocess (bool): if True, the event can be shared between processes, default is False

    Returns:
        Event: The event.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle

            >>> paddle.set_device('custom_cpu')
            >>> e1 = paddle.device.Event()
            >>> e2 = paddle.device.Event('custom_cpu')
            >>> e3 = paddle.device.Event('custom_cpu:0')
            >>> e4 = paddle.device.Event(paddle.CustomPlace('custom_cpu', 0))

    '''

    device: PlaceLike | None
    enable_timing: bool
    event_base: _InitEventBase

    def __init__(
        self,
        device: PlaceLike | None = None,
        enable_timing: bool = False,
        blocking: bool = False,
        interprocess: bool = False,
    ) -> None:
        if device is None:
            self.device = paddle.framework._current_expected_place_()
        elif isinstance(device, str):
            self.device = paddle.device._convert_to_place(device)
        else:
            self.device = device

        if paddle.is_compiled_with_cuda() and isinstance(
            self.device, paddle.CUDAPlace
        ):
            self.event_base = core.CUDAEvent(
                enable_timing, blocking, interprocess
            )
        elif isinstance(self.device, paddle.CustomPlace):
            self.event_base = core.CustomDeviceEvent(
                self.device.get_device_type(),
                self.device.get_device_id(),
                enable_timing,
                blocking,
                interprocess,
            )
        else:
            raise TypeError(
                "device should be gpu, xpu, {}".format(
                    ",".join(paddle.device.get_all_custom_device_type())
                )
            )

    def record(self, stream: Stream | None = None) -> None:
        '''

        Records the event in a given stream.

        Args:
            stream(Stream, optional): The given stream. By default, stream is None,
            event will be recorded in current_stream.

        Returns:
            None.

        Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                >>> import paddle

                >>> paddle.set_device('custom_cpu')
                >>> e = paddle.device.Event()
                >>> e.record()

                >>> s = paddle.device.Stream()
                >>> e.record(s)

        '''
        if stream is None:
            stream = current_stream(self.device)

        self.event_base.record(stream.stream_base)

    def query(self) -> bool:
        '''

        Checks if all work currently captured by event has completed.

        Returns:
            bool: Whether all work currently captured by event has completed.

        Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                >>> import paddle

                >>> paddle.set_device('custom_cpu')
                >>> e = paddle.device.Event()
                >>> e.record()
                >>> e.query()

        '''
        return self.event_base.query()

    def elapsed_time(self, end_event: Event) -> int:
        '''

        Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.

        Returns:
            int: The time.

        Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                >>> import paddle

                >>> paddle.set_device('custom_cpu')
                >>> e1 = paddle.device.Event()
                >>> e1.record()

                >>> e2 = paddle.device.Event()
                >>> e2.record()
                >>> e1.elapsed_time(e2)

        '''
        return self.event_base.elapsed_time(end_event.event_base)

    def synchronize(self) -> None:
        '''

        Waits for the event to complete.
        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.

        Returns:
            None.

        Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                >>> import paddle

                >>> paddle.set_device('custom_cpu')
                >>> e = paddle.device.Event()
                >>> e.record()
                >>> e.synchronize()

        '''
        self.event_base.synchronize()

    def __repr__(self) -> core.CUDAEvent | core.CustomDeviceEvent:
        return self.event_base


class Stream:
    '''

    A device stream wrapper around StreamBase.

    Args:
        device(str|paddle.CUDAPlace(n)|paddle.CustomPlace(n)|None): Which device the stream run on. If device is None, the device is the current device. Default: None.
            It can be ``gpu``, ``gpu:x``, ``custom_device``, ``custom_device:x``, where ``custom_device`` is the name of CustomDevice,
            where ``x`` is the index of the GPUs, XPUs. And it can be paddle.CUDAPlace(n) or paddle.CustomPlace(n).
        priority(int, optional): priority of the CUDA stream. Can be either
            1 (high priority) or 2 (low priority). By default, streams have
            priority 2.

    Returns:
        Stream: The stream.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle

            >>> paddle.set_device('custom_cpu')
            >>> s1 = paddle.device.Stream()
            >>> s2 = paddle.device.Stream('custom_cpu')
            >>> s3 = paddle.device.Stream('custom_cpu:0')
            >>> s4 = paddle.device.Stream(paddle.CustomPlace('custom_cpu', 0))

    '''

    stream_base: _InitStreamBase
    device: PlaceLike

    def __init__(
        self,
        device: PlaceLike | None = None,
        priority: int = 2,
        stream_base: _InitStreamBase | None = None,
    ) -> None:
        if stream_base is not None:
            if isinstance(
                stream_base, (core.CUDAStream, core.CustomDeviceStream)
            ):
                self.stream_base = stream_base
                self.device = stream_base.place
            else:
                raise TypeError(
                    "stream_base should be CUDAStream, CustomDeviceStream"
                )
            return

        if device is None:
            self.device = paddle.framework._current_expected_place_()
        elif isinstance(device, str):
            self.device = paddle.device._convert_to_place(device)
        else:
            self.device = device

        if paddle.is_compiled_with_cuda() and isinstance(
            self.device, paddle.CUDAPlace
        ):
            self.stream_base = core.CUDAStream(
                self.device.get_device_id(), priority
            )
        elif isinstance(self.device, paddle.CustomPlace):
            self.stream_base = core.CustomDeviceStream(
                self.device.get_device_type(),
                self.device.get_device_id(),
                priority,
                blocking=False,
            )
        else:
            raise TypeError(
                "device should be gpu, xpu, {}".format(
                    ",".join(paddle.device.get_all_custom_device_type())
                )
            )

    def wait_event(self, event: Event) -> None:
        '''

        Makes all future work submitted to the stream wait for an event.

        Args:
            event (Event): an event to wait for.

        Returns:
            None.

        Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                >>> import paddle

                >>> paddle.set_device('custom_cpu')
                >>> s1 = paddle.device.Stream()
                >>> s2 = paddle.device.Stream()
                >>> e = paddle.device.Event()
                >>> e.record(s1)
                >>> s2.wait_event(e)

        '''
        self.stream_base.wait_event(event.event_base)

    def wait_stream(self, stream: Stream) -> None:
        '''

        Synchronizes with another stream.
        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Args:
            stream (Stream): a stream to synchronize.

        Returns:
            None.

        Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                >>> import paddle

                >>> paddle.set_device('custom_cpu')
                >>> s1 = paddle.device.Stream()
                >>> s2 = paddle.device.Stream()
                >>> s1.wait_stream(s2)

        '''
        self.stream_base.wait_stream(stream.stream_base)

    def record_event(self, event: Event | None = None) -> Event:
        '''

        Records an event.

        Args:
            event (Event, optional): event to record. If not given, a new one
            will be allocated.

        Returns:
            Event: Recorded event.

        Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                >>> import paddle

                >>> paddle.set_device('custom_cpu')
                >>> s = paddle.device.Stream()
                >>> e1 = s.record_event()

                >>> e2 = paddle.device.Event()
                >>> s.record_event(e2)

        '''
        if event is None:
            event = Event(self.device)
        event.record(self)
        return event

    def query(self) -> bool:
        '''

        Checks if all the work submitted has been completed.

        Returns:
            bool: Whether all kernels in this stream are completed.

        Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                >>> import paddle

                >>> paddle.set_device('custom_cpu')
                >>> s = paddle.device.Stream()
                >>> s.query()

        '''
        return self.stream_base.query()

    def synchronize(self) -> None:
        '''

        Wait for all the kernels in this stream to complete.

        Returns:
            None.

        Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                >>> import paddle

                >>> paddle.set_device('custom_cpu')
                >>> s = paddle.device.Stream()
                >>> s.synchronize()

        '''
        self.stream_base.synchronize()

    @property
    def _as_parameter_(self):
        if isinstance(self.stream_base, core.CUDAStream):
            return ctypes.c_void_p(self.stream_base.cuda_stream)
        else:
            return ctypes.c_void_p(self.stream_base.raw_stream)

    def __eq__(self, o: Stream | None) -> bool:
        if isinstance(o, Stream):
            return super().__eq__(o)
        return False

    def __hash__(self) -> int:
        return hash((self.stream_base, self.device))

    def __repr__(self) -> str:
        return f'<paddle.device.Stream device={self.device} stream={self._as_parameter_.value:#x}>'


def current_stream(device: PlaceLike | None = None) -> Stream:
    '''

    Return the current stream by the device.

    Args:
        device(str|paddle.CUDAPlace(n)|paddle.CustomPlace(n)): The device which want to get stream from.  If device is None, the device is the current device. Default: None.
            It can be ``gpu``, ``gpu:x``, ``custom_device``, ``custom_device:x``, where ``custom_device`` is the name of CustomDevice,
            where ``x`` is the index of the GPUs, CustomDevices. And it can be paddle.CUDAPlace(n) or paddle.CustomPlace(n).

    Returns:
        Stream: The stream to the device.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle

            >>> paddle.set_device('custom_cpu')
            >>> s1 = paddle.device.current_stream()
            >>> s2 = paddle.device.current_stream("custom_cpu:0")
            >>> place = paddle.CustomPlace('custom_cpu', 0)
            >>> s3 = paddle.device.current_stream(place)

    '''
    if device is None:
        place = paddle.framework._current_expected_place_()
    elif isinstance(device, str):
        place = paddle.device._convert_to_place(device)
    else:
        place = device

    if paddle.is_compiled_with_cuda() and isinstance(place, paddle.CUDAPlace):
        return Stream(
            stream_base=core._get_current_stream(place.get_device_id())
        )
    elif isinstance(place, paddle.CustomPlace):
        return Stream(
            stream_base=core._get_current_custom_device_stream(
                place.get_device_type(), place.get_device_id()
            )
        )
    else:
        raise TypeError(
            "device should be gpu, xpu, {}".format(
                ",".join(paddle.device.get_all_custom_device_type())
            )
        )


def set_stream(stream: Stream) -> Stream:
    '''

    Set the current stream.

    Args:
        stream(Stream): The selected stream.

    Returns:
        Stream: The previous stream.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle

            >>> paddle.set_device('custom_cpu')
            >>> s = paddle.device.Stream()
            >>> paddle.device.set_stream(s)

    '''

    prev_stream = current_stream(stream.stream_base.place)

    if paddle.is_compiled_with_cuda() and isinstance(
        stream.stream_base.place, paddle.CUDAPlace
    ):
        core._set_current_stream(stream.stream_base)
    elif isinstance(stream.stream_base.place, paddle.CustomPlace):
        core._set_current_custom_device_stream(
            stream.stream_base.place.get_device_type(),
            stream.stream_base.place.get_device_id(),
            stream.stream_base,
        )
    else:
        raise TypeError(
            "device should be gpu, xpu, {}".format(
                ",".join(paddle.device.get_all_custom_device_type())
            )
        )

    return prev_stream


class stream_guard:
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

            >>> paddle.set_device('custom_cpu')
            >>> s = paddle.device.Stream()
            >>> data1 = paddle.ones(shape=[20])
            >>> data2 = paddle.ones(shape=[20])
            >>> data3 = data1 + data2
            >>> with paddle.device.stream_guard(s):
            ...     s.wait_stream(paddle.device.default_stream()) # type: ignore[attr-defined]
            ...     data4 = data1 + data3

    '''

    stream: Stream | None

    def __init__(self, stream: Stream | None = None) -> None:
        self.stream = stream

    def __enter__(self) -> None:
        cur_stream = self.stream
        if cur_stream is None:
            return

        self.src_prev_stream = current_stream(cur_stream.device)
        if self.src_prev_stream.device != cur_stream.device:
            self.tmp_place = paddle.base.framework._current_expected_place_()
            paddle.base.framework._set_expected_place(cur_stream.device)
            self.dst_prev_stream = current_stream(cur_stream.device)
            set_stream(cur_stream)
        else:
            set_stream(cur_stream)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        cur_stream = self.stream
        if cur_stream is None:
            return

        if self.src_prev_stream.device != cur_stream.device:
            set_stream(self.dst_prev_stream)
            paddle.base.framework._set_expected_place(self.tmp_place)
            set_stream(self.src_prev_stream)
        else:
            set_stream(self.src_prev_stream)


class device_guard:
    '''

    Notes:
        This API only supports dynamic graph mode currently.

    A context manager that specifies the current device context by the given device.

    Args:
        device(PlaceLike): The specified device.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle

            >>> # Set the global default device to CPU
            >>> paddle.set_device("cpu")
            >>> # Temporarily switch to GPU:0 using device_guard with string input
            >>> with paddle.device.device_guard("gpu:0"):
            ...     x = paddle.randn([4, 4])       # Create a Tensor on GPU:0
            ...     x = x.tanh() * 2               # Perform computation on GPU:0
            ...     print(x.place)                 # Check the device of the Tensor
            Place(gpu:0)

            >>> # Set the global default device to GPU:0
            >>> paddle.set_device("gpu:0")
            >>> # Temporarily switch to CPU using device_guard with Place object (CPUPlace)
            >>> cpu_place = paddle.CPUPlace()
            >>> with paddle.device.device_guard(cpu_place):
            ...     x = paddle.randn([4, 4])       # Create a Tensor on CPU
            ...     x = x.tanh() * 2               # Perform computation on CPU
            ...     print(x.place)
            Place(cpu)
    '''

    _target_place: Place
    _original_place: Place

    def __init__(self, device: PlaceLike) -> None:
        if isinstance(device, str):
            self._target_place = paddle.device._convert_to_place(device)
        elif isinstance(device, paddle.base.libpaddle.Place):
            self._target_place = device
        else:
            raise ValueError(
                "'device' must be a string or an instance of a subclass of "
                f"paddle.base.libpaddle.Place, but got {type(device)}"
            )

    def __enter__(self) -> None:
        self._original_place = paddle.framework._current_expected_place_()
        if self._original_place != self._target_place:
            paddle.framework._set_expected_place(self._target_place)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._original_place != self._target_place:
            paddle.framework._set_expected_place(self._original_place)


def synchronize(device: PlaceLike | None = None) -> None:
    """

    Wait for the compute on the given device to finish.

    Args:
        device(str|paddle.CUDAPlace(n)|paddle.XPUPlace(n)|paddle.CustomPlace(n)): The device which want to wait for.  If device is None, the device is the current device. Default: None.
            It can be ``gpu``, ``gpu:x``, ``xpu``, ``xpu:x``, ``custom_device``, ``custom_device:x``, where ``custom_device`` is the name of CustomDevice,
            where ``x`` is the index of the GPUs, XPUs. And it can be paddle.CUDAPlace(n) or paddle.XPUPlace(n) or paddle.CustomPlace(n).

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle

            >>> paddle.set_device('custom_cpu')
            >>> paddle.device.synchronize()
            >>> paddle.device.synchronize("custom_cpu:0")
            >>> place = paddle.CustomPlace('custom_cpu', 0)
            >>> paddle.device.synchronize(place)

    """

    if device is None:
        place = paddle.framework._current_expected_place_()
    elif isinstance(device, str):
        place = paddle.device._convert_to_place(device)
    else:
        place = device

    if paddle.is_compiled_with_cuda() and isinstance(place, paddle.CUDAPlace):
        core._device_synchronize(place.get_device_id())
    elif paddle.is_compiled_with_xpu() and isinstance(place, paddle.XPUPlace):
        core._xpu_device_synchronize(place.get_device_id())
    elif isinstance(place, paddle.CustomPlace):
        core._synchronize_custom_device(
            place.get_device_type(), place.get_device_id()
        )
    else:
        raise TypeError(
            "device should be gpu, xpu, {}".format(
                ",".join(paddle.device.get_all_custom_device_type())
            )
        )
