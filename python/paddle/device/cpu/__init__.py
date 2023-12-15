# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = [
    'max_memory_allocated',
    'max_memory_reserved',
    'memory_allocated',
    'memory_reserved',
]


def max_memory_allocated():
    '''
    Return the peak size of CPU memory that is allocated to tensor of the given device.

    Return:
        int: The peak size of CPU memory that is allocated to tensor of the given device, in bytes.

    Examples:
        .. code-block:: python

            import paddle

            max_memory_allocated_size = paddle.device.cpu.max_memory_allocated()
    '''
    name = "paddle.device.cpu.max_memory_allocated"
    return core.host_memory_stat_peak_value("Allocated", 0)


def max_memory_reserved():
    '''
    Return the peak size of CPU memory that is held by the allocator of the given device.

    Return:
        int: The peak size of CPU memory that is held by the allocator of the given device, in bytes.

    Examples:
        .. code-block:: python

            import paddle

            max_memory_reserved_size = paddle.device.cpu.max_memory_reserved()
    '''
    name = "paddle.device.cpu.max_memory_reserved"
    return core.host_memory_stat_peak_value("Reserved", 0)


def memory_allocated():
    '''
    Return the current size of CPU memory that is allocated to tensor of the given device.

    Return:
        int: The current size of CPU memory that is allocated to tensor of the given device, in bytes.

    Examples:
        .. code-block:: python

            import paddle

            memory_allocated_size = paddle.device.cpu.memory_allocated()
    '''
    name = "paddle.device.cpu.memory_allocated"
    return core.host_memory_stat_current_value("Allocated", 0)


def memory_reserved():
    '''
    Return the current size of CPU memory that is held by the allocator of the given device.

    Return:
        int: The current size of CPU memory that is held by the allocator of the given device, in bytes.

    Examples:
        .. code-block:: python

            import paddle

            memory_reserved_size = paddle.device.cpu.memory_reserved()
    '''
    name = "paddle.device.cpu.memory_reserved"
    return core.host_memory_stat_current_value("Reserved", 0)
