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

from typing import TYPE_CHECKING, TypeAlias, Union

from paddle.base import core

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
        device (str, optional): The device string like 'iluvatar_gpu:0' etc.
            If None, returns properties of the first available custom device with index 0.

    Returns:
        _customDeviceProperties: The properties of the device which include device name,
            major compute capability, minor compute capability, global memory available
            and the number of multiprocessors on the device.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.get_device_properties('iluvatar_gpu:0')
    """
    if device is None:
        dev_types = core.get_all_custom_dev_type()
        if not dev_types:
            raise ValueError("No custom device types available")
        device_name = dev_types[0]
        device_id = 0
    else:
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
                        "Example: 'iluvatar_gpu:0'"
                    )

                device_id = int(device_id_str)
        else:
            raise ValueError(
                f"The input: {device} is not expected. Because paddle.device."
                "get_device_properties only support str or None. "
                "Please input appropriate device again! "
                "Example: 'iluvatar_gpu:0'"
            )

    return core.get_device_properties(device_name, device_id)
