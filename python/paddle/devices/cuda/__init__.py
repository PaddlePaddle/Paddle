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

from paddle.fluid import core, core_avx

from . import streams
from .streams import *

__all__ = [
    'current_stream',
    'synchronize',
]

__all__ += streams.__all__


def current_stream(device=None):
    '''
    Return the current CUDA stream by the device.

    Parameters:
        device(paddle.CUDAPlace()|int, optional): The device or the ID of the device which want to get stream from. 
        If device is None, the device is the current device. Default: None.
    
    Returns:
        CUDAStream: the stream o the device.
    
    Examples:
        .. code-block:: python

            import paddle

            s1 = paddle.devices.cuda.current_stream()

            s2 = paddle.devices.cuda.current_stream(0)

            s3 = paddle.devices.cuda.current_stream(paddle.CUDAPlace(0))

    '''

    device_id = -1

    if device is not None:
        if isinstance(device, int):
            device_id = device
        elif isinstance(device, core.CUDAPlace):
            device_id = device.get_device_id()
        else:
            raise ValueError("device type must be int or paddle.CUDAPlace")

    return core_avx._get_current_stream(device_id)


def synchronize(device=None):
    device_id = -1
    if device is not None:
        if isinstance(device, str):
            device_id = int(device.split(':', 1)[1])
        elif isinstance(device, core.CUDAPlace) or isinstance(device,
                                                              core.XPUPlace):
            device_id = device.get_device_id()

    return core_avx._device_synchronize(device_id)
