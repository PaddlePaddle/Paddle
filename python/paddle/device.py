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

# TODO: define the functions to manipulate devices 
import re
import os
from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.fluid.framework import is_compiled_with_cuda  #DEFINE_ALIAS

__all__ = [
    'get_cudnn_version',
    'set_device',
    'get_device',
    'XPUPlace',
    'is_compiled_with_xpu'
    #            'cpu_places',
    #            'CPUPlace',
    #            'cuda_pinned_places',
    #            'cuda_places',
    #            'CUDAPinnedPlace',
    #            'CUDAPlace',
    'is_compiled_with_cuda'
]

_cudnn_version = None


def is_compiled_with_xpu():
    """
    Whether paddle was built with WITH_XPU=ON to support Baidu Kunlun

    Returns (bool): whether paddle was built with WITH_XPU=ON

    Examples:
        .. code-block:: python

            import paddle
            support_xpu = paddle.device.is_compiled_with_xpu()
    """
    return core.is_compiled_with_xpu()


def XPUPlace(dev_id):
    """
    Return a Baidu Kunlun Place

    Parameters:
        dev_id(int): Baidu Kunlun device id

    Examples:
        .. code-block:: python

            import paddle
            place = paddle.device.XPUPlace(0)
    """
    return core.XPUPlace(dev_id)


def get_cudnn_version():
    """
    This funciton return the version of cudnn. the retuen value is int which represents the 
    cudnn version. For example, if it return 7600, it represents the version of cudnn is 7.6.
    
    Returns:
        int: A int value which represents the cudnn version. If cudnn version is not installed, it return None.

    Examples:
        .. code-block:: python
            
            import paddle

            cudnn_version = paddle.get_cudnn_version()



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


def set_device(device):
    """
    Paddle supports running calculations on various types of devices, including CPU, GPU and XPU.
    They are represented by string identifiers. This function can specify the global device
    which the OP will run.

    Parameters:
        device(str): This parameter determines the specific running device.
            It can be ``cpu``, ``gpu:x`` and ``xpu:x``, where ``x`` is the 
            index of the GPUs or XPUs. 

    Examples:

     .. code-block:: python
            
        import paddle

        paddle.set_device("cpu")
        x1 = paddle.ones(name='x1', shape=[1, 2], dtype='int32')
        x2 = paddle.zeros(name='x2', shape=[1, 2], dtype='int32')
        data = paddle.stack([x1,x2], axis=1)
    """
    lower_device = device.lower()
    if lower_device == 'cpu':
        place = core.CPUPlace()
    elif lower_device == 'gpu':
        if not core.is_compiled_with_cuda():
            raise ValueError(
                "The device should not be 'gpu', " \
                "since PaddlePaddle is not compiled with CUDA")
        place = core.CUDAPlace(ParallelEnv().dev_id)
    elif lower_device == 'xpu':
        if not core.is_compiled_with_xpu():
            raise ValueError(
                "The device should not be 'xpu', " \
                "since PaddlePaddle is not compiled with XPU")
        selected_xpus = os.getenv("FLAGS_selected_xpus", "0").split(",")
        device_id = int(selected_xpus[0])
        place = core.XPUPlace(device_id)
    else:
        avaliable_gpu_device = re.match(r'gpu:\d+', lower_device)
        avaliable_xpu_device = re.match(r'xpu:\d+', lower_device)
        if not avaliable_gpu_device and not avaliable_xpu_device:
            raise ValueError(
                "The device must be a string which is like 'cpu', 'gpu', 'gpu:x', 'xpu' or 'xpu:x'"
            )
        if avaliable_gpu_device:
            if not core.is_compiled_with_cuda():
                raise ValueError(
                    "The device should not be {}, since PaddlePaddle is " \
                    "not compiled with CUDA".format(avaliable_gpu_device))
            device_info_list = device.split(':', 1)
            device_id = device_info_list[1]
            device_id = int(device_id)
            place = core.CUDAPlace(device_id)
        if avaliable_xpu_device:
            if not core.is_compiled_with_xpu():
                raise ValueError(
                    "The device should not be {}, since PaddlePaddle is " \
                    "not compiled with XPU".format(avaliable_xpu_device))
            device_info_list = device.split(':', 1)
            device_id = device_info_list[1]
            device_id = int(device_id)
            place = core.XPUPlace(device_id)
    framework._set_expected_place(place)
    return place


def get_device():
    """
    This funciton can get the current global device of the program is running.
    It's a string which is like 'cpu', 'gpu:x' and 'xpu:x'. if the global device is not
    set, it will return a string which is 'gpu:x' when cuda is avaliable or it 
    will return a string which is 'cpu' when cuda is not avaliable.

    Examples:

     .. code-block:: python
            
        import paddle
        device = paddle.get_device()

    """
    device = ''
    place = framework._current_expected_place()
    if isinstance(place, core.CPUPlace):
        device = 'cpu'
    elif isinstance(place, core.CUDAPlace):
        device_id = place.get_device_id()
        device = 'gpu:' + str(device_id)
    elif isinstance(place, core.XPUPlace):
        device_id = place.get_device_id()
        device = 'xpu:' + str(device_id)

    return device
