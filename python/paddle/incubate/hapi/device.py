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

import six

import paddle.fluid as fluid
from paddle.fluid.dygraph.parallel import ParallelEnv

__all__ = ['set_device', ]

# TODO(qingqing01): remove or refine _global_device, set_device and get_device
# after core framework supporting these function.
_global_device = None


def set_device(device):
    """
    Args:
        device (str): specify device type, 'cpu' or 'gpu'.
        
    Returns:
        fluid.CUDAPlace or fluid.CPUPlace: Created GPU or CPU place.

    Examples:
        .. code-block:: python

        import paddle.incubate.hapi as hapi

        input = hapi.set_device('gpu')
    """

    assert isinstance(device, six.string_types) and device.lower() in ['cpu', 'gpu'], \
    "Expected device in ['cpu', 'gpu'], but got {}".format(device)

    device = fluid.CUDAPlace(ParallelEnv().dev_id) \
            if device.lower() == 'gpu' and fluid.is_compiled_with_cuda() \
                else fluid.CPUPlace()

    global _global_device
    _global_device = device
    return device


def _get_device():
    """
    Return global device.
    """
    if _global_device is not None:
        device = _global_device
    else:
        if fluid.is_compiled_with_cuda():
            device = fluid.CUDAPlace(ParallelEnv().dev_id)
        else:
            device = fluid.CPUPlace()
    return device
