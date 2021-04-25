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

from . import streams
from paddle.fluid import core, core_avx


def current_stream(device=None):
    if device is None:
        device = -1
    return core._get_current_stream(device)


def synchronize(device=None):
    device_id = -1
    if device is not None:
        if isinstance(device, str):
            device_id = int(device.split(':', 1)[1])
        elif isinstance(device, core.CUDAPlace) or isinstance(device,
                                                              core.XPUPlace):
            device_id = device.get_device_id()

    return core_avx._device_synchronize(device_id)
