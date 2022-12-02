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

import paddle
from paddle.fluid import core
from paddle.fluid.core import CustomDeviceEvent as Event  # noqa: F401
from paddle.fluid.core import CustomDeviceStream as Stream  # noqa: F401

__all__ = [
    'get_current_device',
    'set_curren_device',
    'synchronize_device',
    'device_count',
    'current_stream',
    'Event',
    'Stream',
]


def current_device(device_type):
    return core._get_current_custom_device(device_type)


def synchronize_device(place):
    return core._synchronize_custom_device(place)


def device_count(device_type):
    return core._custom_device_count(device_type)


def set_curren_device(place):
    core._set_current_custom_device(place)


def current_stream(place):
    return core._get_current_custom_device_stream(place)
