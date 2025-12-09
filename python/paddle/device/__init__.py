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
import importlib
import os
import re
import sys
import types
from typing import TYPE_CHECKING, Union, overload

from typing_extensions import TypeAlias

import paddle
from paddle.amp import autocast as _autocast
from paddle.base import core, framework
from paddle.base.framework import (
    is_compiled_with_cinn,
    is_compiled_with_cuda,
    is_compiled_with_distribute,
    is_compiled_with_rocm,
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

from . import (  # noqa: F401
    cuda,
    xpu,
)

if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from types import TracebackType

    from paddle import IPUPlace as _IPUPlace, XPUPlace as _XPUPlace
    from paddle._typing.device_like import PlaceLike
    from paddle.base.core import Place

    _InitStreamBase = Union[
        core.CUDAStream, core.CustomDeviceStream, core.XPUStream
    ]
    _InitEventBase = Union[
        core.CUDAEvent, core.CustomDeviceEvent, core.XPUEvent
    ]

    from paddle import CUDAPlace, CustomPlace
    from paddle.base.libpaddle import _customDeviceProperties

    _CustomPlaceLike: TypeAlias = Union[
        CUDAPlace,
        CustomPlace,
        str,  # some string like "iluvatar_gpu" "metax_gpu:0", etc.
        int,  # some int like 0, 1, etc.
    ]

# Dynamically import device functions based on available devices
current_device_is_cpu = 0
if core.is_compiled_with_cuda():
    from .cuda import (
        create_event as _create_event_base,
        create_stream as _create_stream_base,
        device_count,
        empty_cache,
        get_device_properties as _get_device_properties,
        get_rng_state,
        manual_seed,
        max_memory_allocated,
        max_memory_reserved,
        memory_allocated,
        memory_reserved,
        reset_max_memory_allocated,
        reset_max_memory_reserved,
        set_rng_state,
    )
elif core.is_compiled_with_xpu():
    from .xpu import (
        create_event as _create_event_base,
        create_stream as _create_stream_base,
        device_count,
        empty_cache,
        get_rng_state,
        manual_seed,
        max_memory_allocated,
        max_memory_reserved,
        memory_allocated,
        memory_reserved,
        reset_max_memory_allocated,
        reset_max_memory_reserved,
        set_rng_state,
    )
else:
    if hasattr(core, 'get_all_custom_device_type'):
        dev_types = core.get_all_custom_device_type()
    else:
        dev_types = []
    if dev_types and core.is_compiled_with_custom_device(dev_types[0]):
        from .custom_device import (
            create_event as _create_event_base,
            create_stream as _create_stream_base,
            device_count,
            empty_cache,
            get_device_properties as _get_device_properties,
            get_rng_state,
            manual_seed,
            max_memory_allocated,
            max_memory_reserved,
            memory_allocated,
            memory_reserved,
            reset_max_memory_allocated,
            reset_max_memory_reserved,
            set_rng_state,
        )
    else:
        current_device_is_cpu = 1
        from .cpu import (
            device_count,
            get_rng_state,
            manual_seed,
            max_memory_allocated,
            max_memory_reserved,
            reset_max_memory_allocated,
            reset_max_memory_reserved,
            set_rng_state,
        )


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
    'device_count',
    'empty_cache',
    'max_memory_allocated',
    'max_memory_reserved',
    'reset_max_memory_allocated',
    'reset_max_memory_reserved',
    'memory_allocated',
    'memory_reserved',
    'is_available',
    'is_current_stream_capturing',
    'get_device_name',
    'get_device_capability',
    'get_rng_state',
    'set_rng_state',
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
    'device',
    'is_bf16_supported',
    'manual_seed',
    'reset_peak_memory_stats',
    'ipc_collect',
    'get_stream_from_external',
    'StreamContext',
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

            >>> if paddle.device.is_available():
            ...     print("At least one device is available")
            ... else:
            ...     print("No supported devices available")
    """
    return device_count() >= 1


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
            ...     print(paddle.device.is_current_stream_capturing())  # True
            ...     graph.capture_end()
    """
    return core.is_cuda_graph_capturing()


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


def device_to_place(device: Place | int | str | None = None) -> Place:
    """
    Convert input device(Place | int | str | None) into corresponding Place object.
    """
    device = _device_to_paddle(device)
    device = _convert_to_place(device)
    return device


def _convert_to_place(device: PlaceLike) -> Place:
    if not isinstance(device, str):
        return device  # return directly if not a string

    lower_device = device.lower()
    if lower_device.startswith("cuda"):
        lower_device = lower_device.replace("cuda", "gpu")
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


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (paddle.Place, int or str): device index to select.

    Examples:
        .. code-block:: python
            >>> import paddle

            >>> print(paddle.device.get_device())  # gpu:0
            >>> with paddle.device.device("cpu"):
            ...     print(paddle.device.get_device())  # cpu

            >>> # paddle.cuda.device is an alias of paddle.device.device
            >>> with paddle.cuda.device("cpu"):
            ...     print(paddle.device.get_device())  # cpu
            >>> print(paddle.device.get_device())
    """

    def __init__(self, device: Place | int | str | None = None):
        self.place = device_to_place(device)
        self.prev_place_str = "-1"

    def __enter__(self):
        self.prev_place_str = get_device()
        set_device(self.place)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool | None:
        set_device(self.prev_place_str)
        return False


def current_device() -> int:
    """
    Return the index of a currently selected device.

    Returns:
        int: The index of the currently selected device.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> device_id = paddle.device.current_device() # this is equivalent to paddle.cuda.current_device()
            >>> print(f"Current device index: {device_id}")
    """
    # Use paddle.device.get_device() to get the current device string
    device_str = get_device()

    # Parse the device string to extract the device index
    # Format examples: 'gpu:0', 'xpu:0', 'custom_device:0'
    if ':' in device_str:
        device_id = int(device_str.split(':')[1])
    else:
        # If no device index is specified, default to 0
        device_id = 0

    return device_id


def is_bf16_supported(including_emulation: bool = True) -> bool:
    """
    Return a bool indicating if the current CUDA/ROCm device supports dtype bfloat16.

    Args:
        including_emulation (bool = True): Whether to treat software-emulated BF16 as supported; if False, only native hardware BF16 support is considered.

    Returns:
        bool: A boolean value which indicates whether the current CUDA/ROCm device supports dtype bfloat16.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> paddle.device.is_bf16_supported()
            >>> # paddle.cuda.is_bf16_supported() is an alias of paddle.device.is_bf16_supported()
            >>> paddle.cuda.is_bf16_supported()

    """
    # including_emulation is not used here, but kept for compatibility with the original implementation
    if core.is_bfloat16_supported(paddle.framework._current_expected_place_()):
        return True

    # If CUDA is not available, than it does not support bf16 either
    if not is_available():
        return False

    device = get_device()

    # Check for CUDA version and device compute capability.
    # This is a fast way to check for it.
    if not including_emulation:
        return False

    # Finally try to create a bfloat16 device.
    try:
        paddle.tensor([1.0], dtype=paddle.bfloat16, device=device)
        return True
    except:
        return False


def set_device(device: PlaceLike | int) -> PlaceLike:
    """

    Paddle supports running calculations on various types of devices, including CPU, GPU, XPU, NPU and IPU.
    They are represented by string identifiers. This function can specify the global device
    which the OP will run.

    Args:
        device(str, Place or int): This parameter determines the specific running device.
            It can be ``cpu``, ``gpu``, ``xpu``, ``npu``, ``gpu:x``, ``xpu:x``, ``npu:x`` and ``ipu``,
            where ``x`` is the index of the GPUs, XPUs or NPUs.

    Returns:
        Place,the Place to set.

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle

            >>> paddle.device.set_device("cpu")
            >>> x1 = paddle.ones(name='x1', shape=[1, 2], dtype='int32')
            >>> print(x1.place)
            Place(cpu)

            >>> paddle.device.set_device("gpu:0")
            >>> x2 = paddle.zeros(name='x2', shape=[1, 2], dtype='int32')
            >>> print(x2.place)
            Place(gpu:0)
            >>> # x1 is still on cpu
            >>> print(x1.place)
            Place(cpu)

    """
    place = device_to_place(device)
    framework._set_expected_place(place)
    return place


@overload
def get_device(input: None = None) -> str: ...


@overload
def get_device(input: paddle.Tensor) -> int: ...


def get_device(input: paddle.Tensor | None = None) -> str | int:
    """

    This function can get the current global device of the program is running.
    It's a string which is like 'cpu', 'gpu:x', 'xpu:x' and 'npu:x'. if the global device is not
    set, it will return a string which is 'gpu:x' when cuda is available or it
    will return a string which is 'cpu' when cuda is not available.

    Returns:
        if input is Tensor, this function will return the device ID where the given Tensor is located.
        int:
            - -1, if the Tensor is on CPU.
            - The device ID (e.g., 0, 1, ...) if the Tensor is on GPU.

        if input is not Tensor, this function will return the device name where the program is running.
        str:
            - 'cpu': If the program is running on CPU.
            - 'gpu:x': If the program is running on GPU, where `x` is the index of the GPU.
            - 'xpu:x': If the program is running on XPU, where `x` is the index of the XPU.
            - 'npu:x': If the program is running on NPU, where `x` is the index of
    Examples:

        .. code-block:: python

            >>> import paddle
            >>> device = paddle.device.get_device()

            >>> x_cpu = paddle.to_tensor([1, 2, 3], place=paddle.CPUPlace())
            >>> id = paddle.get_device(x_cpu) # -1



    """
    if isinstance(input, paddle.Tensor):
        if 'cpu' in str(input.place):
            return -1
        return input.place.gpu_device_id()
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


def get_default_device() -> paddle.device:
    """
    Returns:
        str: The default device for PaddlePaddle.
    Example:
        .. code-block:: pycon

            >>> import paddle

            >>> print(paddle.get_default_device())
    """
    return paddle.device(get_device().replace("gpu", "cuda"))


def set_default_device(device: PlaceLike | int) -> None:
    """
    Paddle supports running calculations on various types of devices, including CPU, GPU, XPU, NPU and IPU.
    This function can specify the global device which the OP will run.

    Args:
        device(str, Place or int): This parameter determines the specific running device.
            It can be ``cpu``, ``gpu``, ``xpu``, ``npu``, ``gpu:x``, ``xpu:x``, ``npu:x`` and ``ipu``,
            where ``x`` is the index of the GPUs, XPUs or NPUs.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.set_device("cpu")
    """
    set_device(device)


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
            >>> # Output: []

            >>> # Case 2: paddlepaddle-gpu package installed, and no custom device registered.
            >>> # Output: ['gpu']

            >>> # Case 3: paddlepaddle-cpu package installed, and custom device 'CustomCPU' is registered.
            >>> # Output: ['CustomCPU']

            >>> # Case 4: paddlepaddle-gpu package installed, and custom device 'CustomCPU' and 'CustomGPU' is registered.
            >>> # Output: ['gpu', 'CustomCPU', 'CustomGPU']

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
            >>> # Output: []

            >>> # Case 2: paddlepaddle-gpu package installed, and no custom device registered.
            >>> # Output: ['gpu:0', 'gpu:1']

            >>> # Case 3: paddlepaddle-cpu package installed, and custom device 'CustomCPU' is registered.
            >>> # Output: ['CustomCPU']

            >>> # Case 4: paddlepaddle-gpu package installed, and custom device 'CustomCPU' and 'CustomGPU' is registered.
            >>> # Output: ['gpu:0', 'gpu:1', 'CustomCPU', 'CustomGPU:0', 'CustomGPU:1']

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
    device = _device_to_paddle(device)
    return _get_device_properties(device)


def get_device_module(device: _CustomPlaceLike = None):
    """
    Returns the Paddle module associated with a given device.

    Args:
        device (_CustomPlaceLike, optional): The device to query.
            Can be one of the following:
                - paddle.Place object (e.g., paddle.CUDAPlace(0))
                - str (e.g., "gpu:0", "xpu", "npu")
                - int (device index, e.g., 0 -> "gpu:0")
                - None (use current expected place)

    Returns:
        module: The corresponding Paddle device module (e.g., paddle.cuda, paddle.device.xpu)

    Raises:
        RuntimeError: If the device type is CPU (Paddle does not expose `paddle.cpu`)
                      or if no matching device module is found.

    Example:
        .. code-block:: python
        >>> paddle.get_device_module("gpu:0")
        <module 'paddle.cuda' ...>

        >>> # paddle.get_device_module(paddle.XPUPlace(0))
        >>> # <module 'paddle.device.xpu' ...>
    """
    device = _device_to_paddle(device)
    if isinstance(device, str):
        device = device.lower().split(':')[0]
        custom_device_types = {
            "metax_gpu",
            "biren_gpu",
            "custom_cpu",
            "gcu",
            "iluvatar_gpu",
            "intel_gpu",
            "intel_hpu",
            "mlu",
            "mps",
            "npu",
            "sdaa",
        }
        if device in ("cuda", "gpu"):
            return paddle.cuda
        elif device == "xpu":
            return paddle.device.xpu
        elif device in custom_device_types:
            return paddle.device.custom_device
        elif device == "cpu":
            return paddle.device.cpu
        else:
            raise RuntimeError(f"Unsupported device type: {device}")

    place = (
        paddle.framework._current_expected_place_()
        if device is None
        else _convert_to_place(device)
    )

    place_to_module = {
        core.CUDAPlace: paddle.cuda,
        core.CustomPlace: paddle.device.custom_device,
        core.XPUPlace: paddle.device.xpu,
        core.CPUPlace: paddle.device,
    }

    for place_type, module in place_to_module.items():
        if isinstance(place, place_type):
            return module


def get_device_name(
    device: _CustomPlaceLike | None = None,
) -> str:
    """

    Return the properties of given device.

    Args:
        device(|paddle.CustomPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like npu:x' which to get the properties of the
            device from. If device is None, the device is the current device.
            Default: None.

    Returns:
        str: The name of the CUDA device.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle
            >>> name = paddle.device.get_device_name()
            >>> print(name)
    """
    return get_device_properties(device).name


def get_device_capability(
    device: _CustomPlaceLike | None = None,
) -> tuple[int, int]:
    """

    Return the device_capability of given device.

    Args:
        device(|paddle.CustomPlace|int|str|None, optional): The device, the id of the device or
            the string name of device like npu:x' which to get the properties of the
            device from. If device is None, the device is the current device.
            Default: None.

    Returns:
        str: The device_capability of given device.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle
            >>> cap = paddle.device.get_device_capability()
            >>> print(cap)
    """
    prop = get_device_properties(device)
    return prop.major, prop.minor


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


class Event:
    '''

    A device event wrapper around StreamBase.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time, default is False
        blocking (bool, optional): if True, ``wait`` will be blocking, default is False
        interprocess (bool): if True, the event can be shared between processes, default is False

    Returns:
        Event: The event.

    Note:
        The `device` parameter has been removed in the latest version. The event will always use the current device context.
        Previously, you could specify the device like:
        ```python
        # Old usage (no longer supported)
        e = paddle.device.Event(device="gpu:0")
        ```
        Now it will automatically use the current device:
        ```python
        # New usage
        paddle.set_device("gpu:0")  # Set device first
        e = paddle.device.Event()  # Will use gpu:0
        ```

        paddle.device.Event is equivalent to paddle.cuda.Event.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle

            >>> paddle.set_device('custom_cpu')
            >>> e1 = paddle.device.Event()  # Uses current device (custom_cpu)
            >>>
            >>> # Old usage (no longer supported):
            >>> # e2 = paddle.device.Event('custom_cpu')
            >>> # e3 = paddle.device.Event('custom_cpu:0')
            >>> # e4 = paddle.device.Event(paddle.CustomPlace('custom_cpu', 0))
            >>>
            >>> # New equivalent usage:
            >>> paddle.set_device('custom_cpu:0')
            >>> e5 = paddle.device.Event()  # Uses custom_cpu:0

    '''

    device: PlaceLike | None
    enable_timing: bool
    event_base: _InitEventBase

    def __init__(
        self,
        enable_timing: bool = False,
        blocking: bool = False,
        interprocess: bool = False,
    ) -> None:
        self.device = paddle.framework._current_expected_place_()

        device_id = (
            self.device.get_device_id()
            if hasattr(self.device, 'get_device_id')
            else None
        )
        device_type = (
            self.device.get_device_type()
            if hasattr(self.device, 'get_device_type')
            else None
        )

        self.event_base = _create_event_base(
            enable_timing=enable_timing,
            blocking=blocking,
            interprocess=interprocess,
            device_type=device_type,
            device_id=device_id,
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

    def __repr__(self) -> str:
        return f"Event(device={self.device}, event_base={self.event_base})"


class Stream:
    '''

    A device stream wrapper around StreamBase.
    paddle.cuda.Stream() is equivalent to paddle.device.Stream().

    Args:
        device(str|paddle.CUDAPlace(n)|paddle.CustomPlace(n)|None): Which device the stream run on. If device is None, the device is the current device. Default: None.
            It can be ``gpu``, ``gpu:x``, ``custom_device``, ``custom_device:x``, where ``custom_device`` is the name of CustomDevice,
            where ``x`` is the index of the GPUs, XPUs. And it can be paddle.CUDAPlace(n) or paddle.CustomPlace(n).
        priority(int, optional): priority of the CUDA stream. Can be either
            1 or -1 (high priority) or 0 or 2 (low priority). By default, streams have
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
    device: PlaceLike | int
    _priority_map: dict[int, int] = {-1: 1, 0: 2, 1: 1, 2: 2}

    def __init__(
        self,
        device: PlaceLike | int | None = None,
        priority: int = 2,
        stream_base: _InitStreamBase | None = None,
    ) -> None:
        if stream_base is not None:
            if isinstance(
                stream_base,
                (core.CUDAStream, core.CustomDeviceStream, core.XPUStream),
            ):
                self.stream_base = stream_base
                self.device = stream_base.place
            else:
                raise TypeError(
                    "stream_base should be CUDAStream, XPUStream, CustomDeviceStream"
                )
            return
        self.device = device_to_place(device)

        device_id = (
            self.device.get_device_id()
            if hasattr(self.device, 'get_device_id')
            else None
        )
        device_type = (
            self.device.get_device_type()
            if hasattr(self.device, 'get_device_type')
            else None
        )
        priority = self._priority_map.get(priority, 2)
        self.stream_base = _create_stream_base(
            device_id=device_id,
            priority=priority,
            blocking=False,
            device_type=device_type,
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
            event = Event()
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
        elif isinstance(self.stream_base, core.XPUStream):
            return ctypes.c_void_p(self.stream_base.xpu_stream)
        else:
            return ctypes.c_void_p(self.stream_base.raw_stream)

    def __cuda_stream__(self):
        """
        CUDA Stream protocol described at
        https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol

        Returns a tuple of (protocol_version, cudaStream_t)
        """
        return (0, self.stream_base.raw_stream)

    def __eq__(self, o: Stream | None) -> bool:
        if isinstance(o, Stream):
            return super().__eq__(o)
        return False

    def __hash__(self) -> int:
        return hash((self.stream_base, self.device))

    def __repr__(self) -> str:
        return f'<paddle.device.Stream device={self.device} stream={self._as_parameter_.value:#x}>'


def _device_to_paddle(
    dev: Place | int | str | None = None,
):
    if isinstance(dev, int):
        if dev < 0:
            raise ValueError(f"Device index must be non-negative, got {dev}")
        current_place = get_device()  # e.g. "gpu:0", "cpu"
        if current_place == "cpu":
            if dev != 0:
                raise ValueError(f"CPU device only supports index 0, got {dev}")
            return "cpu"
        device_type = current_place.split(":")[0]
        return f"{device_type}:{dev}"
    elif isinstance(dev, str):
        cleaned_device = dev.strip()
        return (
            cleaned_device.replace("cuda:", "gpu:")
            if "cuda:" in cleaned_device
            else cleaned_device
        )
    elif dev is None:
        return get_device()
    else:
        return dev


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
    elif paddle.is_compiled_with_xpu() and isinstance(place, paddle.XPUPlace):
        return Stream(
            stream_base=core._xpu_get_current_stream(place.get_device_id())
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
            >>> # paddle.cuda.set_stream(s) is equivalent to paddle.device.set_stream(s)
            >>> paddle.device.set_stream(s)

    '''

    prev_stream = current_stream(stream.stream_base.place)

    if paddle.is_compiled_with_cuda() and isinstance(
        stream.stream_base.place, paddle.CUDAPlace
    ):
        core._set_current_stream(stream.stream_base)
    elif paddle.is_compiled_with_xpu() and isinstance(
        stream.stream_base.place, paddle.XPUPlace
    ):
        core._xpu_set_current_stream(stream.stream_base.idx)
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
            >>> with paddle.device.stream_guard(s):# this is equivalent to paddle.cuda.StreamContext(s) and paddle.device.StreamContext(s)
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


StreamContext = stream_guard


def stream(stream: Stream | None) -> stream_guard:
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
            >>> s = paddle.device.Stream()
            >>> data1 = paddle.ones(shape=[20])
            >>> data2 = paddle.ones(shape=[20])
            >>> data3 = data1 + data2

            >>> with paddle.device.stream(s): # this is equivalent to paddle.cuda.stream(s)
            ...     s.wait_stream(paddle.cuda.current_stream())
            ...     data4 = data1 + data3
            >>> print(data4)

    '''
    return StreamContext(stream)


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


def ipc_collect() -> None:
    """
    Force collects GPU memory after it has been released by CUDA IPC.
    This function checks if any sent CUDA tensors could be cleaned from the memory.
    Force closes shared memory file used for reference counting if there is no active counters.
    Useful when the producer process stopped actively sending tensors and want to release unused memory.
    Returns:
        None
    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> # Force collect expired IPC memory
            >>> paddle.device.ipc_collect() #this is equivalent to paddle.cuda.ipc_collect()
    """
    paddle.base.libpaddle._ipc_collect()


def get_stream_from_external(
    data_ptr: int, device: PlaceLike | None = None
) -> Stream:
    r'''
    Return a :class:`Stream` from an externally allocated CUDA stream.

    This function is used to wrap streams allocated in other libraries in order
    to facilitate data exchange and multi-library interactions.

    .. note::
        This function doesn't manage the stream life-cycle, it is the user
        responsibility to keep the referenced stream alive while this returned
        stream is being used.

    Args:
        data_ptr(int): Integer representation of the CUDA stream handle (``cudaStream_t``)
            that is allocated externally.
        device(str|paddle.CUDAPlace(n), optional):
            The CUDA device where the stream was originally allocated.
            If device is None, the current CUDA device is used.
            It can be ``gpu``, ``gpu:x``, or ``paddle.CUDAPlace(n)``.

    Returns:
        Stream: The wrapped CUDA stream corresponding to the given external pointer.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> # doctest: +SKIP('original_raw_ptr not exist')
            >>> original_raw_ptr = 77777
            >>> external_stream = paddle.device.get_stream_from_external(original_raw_ptr,"cuda:0")
    '''
    if device is None:
        place = paddle.framework._current_expected_place_()
    elif isinstance(device, str):
        place = paddle.device._convert_to_place(device)
    else:
        place = device

    return Stream(
        stream_base=core._get_stream_from_external(
            data_ptr, place.get_device_id()
        )
    )


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
            >>> paddle.device.manual_seed_all(102)

    """
    paddle.seed(seed)


class _AutocastMode:
    @staticmethod
    def autocast(
        enabled=True, dtype=paddle.float16, cache_enabled=True
    ) -> AbstractContextManager:
        """
        Create a context which enables auto-mixed-precision(AMP) of operators executed in dynamic graph mode.
        If enabled, the input data type (float32, float16 or bfloat16) of each operator is decided
        by autocast algorithm for better performance.

        Commonly, it is used together with `GradScaler` and `decorator` to achieve Auto-Mixed-Precision in
        imperative mode.

        Args:
            device_type(str, optional): Device type. But because the paddle does not distinguish between devices, this parameter does not work.
            enable(bool, optional): Enable auto-mixed-precision or not. Default is True.
            dtype(str, optional): Whether to use 'float16' or 'bfloat16'. Default is 'float16'.
            cache_enabled(bool, optional): whether to enable cache or not. Default is True. But this parameter is not used

        Note:
            paddle.cuda.amp.

        Examples:

            .. code-block:: python

                >>> # doctest: +REQUIRES(env:GPU)
                >>> import paddle

                >>> conv2d = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
                >>> data = paddle.rand([10, 3, 32, 32])

                >>> with paddle.device.amp.auto_cast():
                ...     conv = conv2d(data)
                ...     print(conv.dtype)
                >>> # doctest: +SKIP("This has diff in xdoctest env")
                paddle.float16
                >>> # doctest: -SKIP

                >>> with paddle.device.amp.auto_cast(enable=False):
                ...     conv = conv2d(data)
                ...     print(conv.dtype)
                >>> # doctest: +SKIP("This has diff in xdoctest env")
                paddle.float32
                >>> # doctest: -SKIP

        """
        return _autocast(device_type='cuda', enabled=enabled, dtype=dtype)


class amp:
    """Namespace for amp marker operations."""

    autocast = staticmethod(_AutocastMode.autocast)
    autocast_mode = _AutocastMode()


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
                >>> paddle.device.nvtx.range_push("test")

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
                >>> paddle.device.nvtx.range_pop()
        """
        paddle.base.core.nvprof_nvtx_pop()


def reset_peak_memory_stats(device: PlaceLike | int | None = None) -> None:
    """
    Resets all devices' peak memory statistics.

    This method resets the peak memory usage recorded for each device during the execution of the program.
    It sets the peak memory usage back to zero for all devices.

    Example:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')  # or '<custom_device>'

            >>> # paddle.cuda.reset_max_memory_allocated() is equivalent to paddle.device.reset_max_memory_allocated()

            >>> paddle.device.reset_max_memory_allocated(paddle.CUDAPlace(0))
            >>> paddle.device.reset_max_memory_allocated(0)
            >>> paddle.device.reset_max_memory_allocated("gpu:0")
    """
    reset_max_memory_allocated()


class Device(str):
    """
    Paddle computing device.

    This class represents a computing device in Paddle, such as CPU, GPU (CUDA), or XPU,
    and can be passed directly to Paddle tensor creation APIs.

    Note:
        - Only device types "cpu", "gpu", "cuda", and "xpu" are supported.
        - The string representation of the device (e.g., "cuda:0") can be used directly
          in Paddle APIs that accept a device argument.
        - This class supports context manager usage to temporarily set the default device.

    Args:
        type (str|int, optional): The device type or a legacy device index.
            - str: "cpu", "cuda", "cuda:0", "gpu:1", "xpu:0"
            - int: legacy, interpreted as the default GPU device index
        index (int, optional): The device index, used with `type` string. Ignored for CPU.

    Attributes:
        type (str): Device type ("cpu", "cuda", "gpu", "xpu").
        index (int|None): Device index. None for CPU.

    Examples:
        .. code-block:: python

            >>> import paddle

            # String initialization
            >>> d1 = paddle.device("cpu")
            >>> d2 = paddle.device("cuda:0")
            >>> d3 = paddle.device("xpu", 1)

            # Type + index initialization
            >>> d4 = paddle.device(type="cuda", index=0)

            # Legacy int initialization
            >>> d5 = paddle.device(0)  # equivalent to paddle.device("cuda", 0)

            # Copy from another device
            >>> d6 = paddle.device(d2)

            # Using as context manager
            >>> with paddle.device("cuda:1"):
            ...     x = paddle.zeros([2, 3])  # created on CUDA device 1

            >>> print(d2.type)   # "cuda"
            >>> print(d2.index)  # 0
            >>> print(d1)        # "cpu"
            >>> print(d2)        # "cuda:0"
    """

    _DEFAULT_DEVICE_STACK = []
    _SUPPORTED_TYPES = {"cpu", "gpu", "cuda", "xpu"}

    def __new__(
        cls, type: PlaceLike | int | None = None, index: int | None = None
    ):
        if isinstance(type, paddle.base.libpaddle.Place):
            if type.is_cpu_place():
                dev_type = 'cpu'
                dev_index = None
            elif type.is_gpu_place():
                dev_type = 'cuda'
                dev_index = type.gpu_device_id()
            elif type.is_xpu_place():
                dev_type = 'xpu'
                dev_index = type.gpu_device_id()
            elif type.is_custom_place():
                dev_type = type.get_device_type()
                dev_index = type.get_device_id()
            else:
                raise ValueError(f"Unknown place type: {type}")

        elif isinstance(type, str):
            t = type.lower()
            if t not in cls._SUPPORTED_TYPES and ":" not in t:
                raise ValueError(f"Unsupported device type: {t}")
            if index is not None:
                dev_type = t
                dev_index = index if t != "cpu" else None
            else:
                if ":" in t:
                    dev_type, idx = t.split(":")
                    dev_type = dev_type.lower()
                    if dev_type not in cls._SUPPORTED_TYPES:
                        raise ValueError(f"Unsupported device type: {dev_type}")
                    dev_index = int(idx)
                else:
                    dev_type = t
                    dev_index = 0 if t != "cpu" else None

        elif isinstance(type, int):
            dev_type = "cuda"
            dev_index = type

        elif type is None and index is not None:
            raise ValueError("Device type must be specified if index is given")

        else:
            raise TypeError(f"Unsupported type for Device: {type}")

        s = f"{dev_type}:{dev_index}" if dev_type != "cpu" else "cpu"
        obj = str.__new__(cls, s)
        obj._dev_type = dev_type
        obj._index = dev_index
        return obj

    @property
    def type(self):
        return self._dev_type

    @property
    def index(self):
        return self._index

    def _to_place(self) -> core.Place:
        if self.type == "cpu":
            return core.CPUPlace()
        elif self.type in {"gpu", "cuda"}:
            return core.CUDAPlace(self.index)
        elif self.type == "xpu":
            return core.XPUPlace(self.index)
        else:
            raise ValueError(f"Unsupported device type: {self.type}")

    def __dlpack_device__(self) -> tuple[int, int]:
        return self._to_place().__dlpack_device__()

    def __enter__(self):
        current_device = paddle.get_device()
        Device._DEFAULT_DEVICE_STACK.append(current_device)
        paddle.set_device(str(self))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        previous_device = Device._DEFAULT_DEVICE_STACK.pop()
        paddle.set_device(previous_device)


class _DeviceModule(types.ModuleType):
    """A callable package module: paddle.device(...) -> Device(...)"""

    def __call__(self, *args, **kwargs) -> Device:
        return Device(*args, **kwargs)

    def __getattr__(self, name: str):
        # support lazy import submodeule：paddle.device.cuda / paddle.device.xpu / ...
        try:
            mod = importlib.import_module(f"{self.__name__}.{name}")
            setattr(self, name, mod)
            return mod
        except ModuleNotFoundError as e:
            raise AttributeError(name) from e


_self = sys.modules[__name__]
_proxy = _DeviceModule(__name__, _self.__doc__)
_proxy.__dict__.update(_self.__dict__)
sys.modules[__name__] = _proxy
