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

if TYPE_CHECKING:
    from paddle import CustomPlace
    from paddle.base.libpaddle import _customDeviceProperties

    _CustomPlaceLike: TypeAlias = Union[
        CustomPlace,
        str,
        int,
    ]

__all__ = [
    'get_device_properties',
]


def get_device_properties(
    device: _CustomPlaceLike | None = None,
) -> _customDeviceProperties:
    """
    Return the properties of given custom device.

    Args:
        device (CustomPlace|str|int|None, optional): The device, the id of the device or
            the string name of device like 'metax_gpu:x' which to get the properties of the
            device from. Notice that this api only supports gpgpu devices. If device is None, the device is the current device.
            Default: None.

    Returns:
        _customDeviceProperties: The properties of the device which include device name,
            major compute capability, minor compute capability, global memory available
            and the number of multiprocessors on the device.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.get_device_properties('metax_gpu:0')
            >>> paddle.device.get_device_properties(0)
            >>> paddle.device.get_device_properties(paddle.CustomPlace('metax_gpu', 0))
    """
    raise RuntimeError(
        "get_device_properties is not supported for this device type. "
        "This function is only available for gpgpu devices."
    )
    return None
