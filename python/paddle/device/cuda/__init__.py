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

from paddle.fluid import core

from .streams import Stream  # noqa: F401
from .streams import Event  # noqa: F401

__all__ = [
    'Stream',
    'Event',
    'current_stream',
    'synchronize',
    'device_count',
    'empty_cache',
    'get_device_properties'
]


def current_stream(device=None):
    '''
    Return the current CUDA stream by the device.

    Parameters:
        device(paddle.CUDAPlace()|int, optional): The device or the ID of the device which want to get stream from. 
        If device is None, the device is the current device. Default: None.
    
    Returns:
        CUDAStream: the stream to the device.
    
    Examples:
        .. code-block:: python

            # required: gpu
            import paddle

            s1 = paddle.device.cuda.current_stream()

            s2 = paddle.device.cuda.current_stream(0)

            s3 = paddle.device.cuda.current_stream(paddle.CUDAPlace(0))

    '''

    device_id = -1

    if device is not None:
        if isinstance(device, int):
            device_id = device
        elif isinstance(device, core.CUDAPlace):
            device_id = device.get_device_id()
        else:
            raise ValueError("device type must be int or paddle.CUDAPlace")

    return core._get_current_stream(device_id)


def synchronize(device=None):
    '''
    Wait for the compute on the given CUDA device to finish.

    Parameters:
        device(paddle.CUDAPlace()|int, optional): The device or the ID of the device.
        If device is None, the device is the current device. Default: None.
    
    Examples:
        .. code-block:: python

            # required: gpu
            import paddle

            paddle.device.cuda.synchronize()
            paddle.device.cuda.synchronize(0)
            paddle.device.cuda.synchronize(paddle.CUDAPlace(0))

    '''

    device_id = -1

    if device is not None:
        if isinstance(device, int):
            device_id = device
        elif isinstance(device, core.CUDAPlace):
            device_id = device.get_device_id()
        else:
            raise ValueError("device type must be int or paddle.CUDAPlace")

    return core._device_synchronize(device_id)


def device_count():
    '''
    Return the number of GPUs available.
    
    Returns:
        int: the number of GPUs available.

    Examples:
        .. code-block:: python

            import paddle

            paddle.device.cuda.device_count()

    '''

    num_gpus = core.get_cuda_device_count() if hasattr(
        core, 'get_cuda_device_count') else 0

    return num_gpus

def empty_cache():
    """
    Releases idle cached memory held by the allocator so that those can be used in other GPU
    application and visible in `nvidia-smi`. In most cases you don't need to use this function,
    Paddle does not release the memory back to the OS when you remove Tensors on the GPU,
    Because it keeps gpu memory in a pool so that next allocations can be done much faster.

    Examples:
        .. code-block:: python

            import paddle

            # required: gpu
            paddle.set_device("gpu")
            tensor = paddle.randn([512, 512, 512], "float")
            del tensor
            paddle.device.cuda.empty_cache()
    """

    if core.is_compiled_with_cuda():
        core.cuda_empty_cache()


def get_device_properties(device):
    '''
    Return the properties of given CUDA device.

    Args:
        device(paddle.CUDAPlace() or int): The device or the ID of the device which want to get the properties of the device from. 

    Returns:
        _CudaDeviceProperties: the properties of the device which include ASCII string identifying device, major compute capability, minor compute capability, 
                               global memory available on device in bytes and the number of multiprocessors on the device.

    Examples:
    
        .. code-block:: python

            # required: gpu
            import paddle

            paddle.device.cuda.get_device_properties(0)
            paddle.device.cuda.get_device_properties(paddle.CUDAPlace(0))

    '''
    device_id = -1
    if device is not None:
        if isinstance(device, int):
            device_id = device
        elif isinstance(device, core.CUDAPlace):
            device_id = device.get_device_id()
        else:
            raise ValueError("device type must be int or paddle.CUDAPlace")

    return core.get_device_properties(device_id)
