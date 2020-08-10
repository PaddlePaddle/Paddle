#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid import core
from paddle.fluid import framework
import re
__all__ = [
    'set_device',
    'get_device'
    #            'cpu_places',
    #            'CPUPlace',
    #            'cuda_pinned_places',
    #            'cuda_places',
    #            'CUDAPinnedPlace',
    #            'CUDAPlace',
    #            'is_compiled_with_cuda'
]


def set_device(device):
    """
    This function can determine whether the program is running on the CPU
    or GPU place.
    Parameters:
        device(str): This parameter determines the specific running device.
            It can be ``cpu`` or ``gpu:0``. When ``device`` is ``cpu``, the
            program is running on the cpu. When ``device`` is ``gpu``, the
            program is running ont the gpu.
    Examples:

     .. code-block:: python
            
        import paddle
        paddle.enable_imperative()
        paddle.fluid.dygraph.set_device("gpu:0")
        x1 = paddle.ones(name='x1', shape=[1, 2], dtype='int32')
        x2 = paddle.zeros(name='x2', shape=[1, 2], dtype='int32')
        data = paddle.stack([x1,x2], axis=1)
    """
    lower_device = device.lower()
    if lower_device == 'cpu':
        place = core.CPUPlace()
        framework._set_expected_place(place)
        framework._set_dygraph_tracer_expected_place(place)
    else:
        avaliable_device = ((lower_device == 'cpu') or
                            re.match(r'gpu:\d+', lower_device))
        if not avaliable_device:
            raise ValueError(
                "The device must be a string which is like 'cpu' or 'gpu:0'")
        device_info_list = device.split(':', 1)
        device_id = device_info_list[1]
        device_id = int(device_id)
        core.set_device_id(device_id)
        place = core.CUDAPlace(device_id)
        framework._set_expected_place(place)
        framework._set_dygraph_tracer_expected_place(place)


def get_device():
    """
    This funciton can get the device which is the programming is running.

    Examples:

     .. code-block:: python
            
        import paddle
        paddle.enable_imperative()
        device = paddle.fluid.dygraph.get_device()

    """
    device = ''
    place = framework._current_expected_place()
    print(place)
    if isinstance(place, core.CPUPlace):
        device = 'cpu'
    else:
        device_id = core.get_device_id()
        device = 'gpu:' + str(device_id)

    return device
