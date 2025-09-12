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

# from typing import TYPE_CHECKING
from paddle.base import core

# if TYPE_CHECKING:
#     from paddle import CustomPlace

dev_types = core.get_all_custom_device_type()
if not dev_types:
    raise ValueError(
        "No custom device available, please install paddle with custom device support"
    )

dev_type = dev_types[0]

if not core.is_compiled_with_custom_device(dev_type):
    raise Exception(
        "No custom device available, please install paddle with custom device support"
    )

if dev_type in ['metax_gpu', 'iluvatar_gpu']:
    from .gpgpu_backend import get_device_properties

__all__ = [
    'device_count',
    'get_device_properties',
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
