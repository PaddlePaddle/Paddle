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

from paddle.base import core
from paddle.utils import deprecated

__all__ = [
    'synchronize',
]


@deprecated(
    since="2.5.0",
    update_to="paddle.device.synchronize",
    level=1,
    reason="synchronize in paddle.device.xpu will be removed in future",
)
def synchronize(device=None):
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


def device_count():
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


def set_debug_level(level=1):
    '''
    Set the debug level of XPUs' api.

    Parameters:
        int: debug level of XPUs available.
        |level        |name       |usage
        |0            |stop       |stop the debug mode
        |0x1          |trace      |Print the invocation of the interface
        |0x10         |checksum   |Print the checksum of the tensor
        |0x100        |dump       |Save the tensor as a file in npy format
        |0x1000       |profiling  |Record the execution time of each operator

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.device.xpu.set_debug_level(0x1)
    '''
    core.set_xpu_debug_level(level)
