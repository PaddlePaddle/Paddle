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

import paddle
from paddle.fluid import core

__all__ = [
    'synchronize',
]


def synchronize(device=None):
    '''
    Wait for the compute on the given XPU device to finish.

    Parameters:
        device(paddle.XPUPlace()|int, optional): The device or the ID of the device.
        If device is None, the device is the current device. Default: None.

    Examples:
        .. code-block:: python

            # required: xpu
            import paddle

            paddle.device.xpu.synchronize()
            paddle.device.xpu.synchronize(0)
            paddle.device.xpu.synchronize(paddle.XPUPlace(0))

    '''

    device_id = -1

    if device is not None:
        if isinstance(device, int):
            device_id = device
        elif isinstance(device, core.XPUPlace):
            device_id = device.get_device_id()
        else:
            raise ValueError("device type must be int or paddle.XPUPlace")

    return core._xpu_device_synchronize(device_id)
