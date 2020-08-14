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
from paddle.fluid import core
__all__ = [
    'get_cudnn_version',
    #            'cpu_places',
    #            'CPUPlace',
    #            'cuda_pinned_places',
    #            'cuda_places',
    #            'CUDAPinnedPlace',
    #            'CUDAPlace',
    #            'is_compiled_with_cuda'
]

_cudnn_version = None


def get_cudnn_version():
    """
    This funciton return the version of cudnn.
    """
    global _cudnn_version
    if _cudnn_version is None:
        cudnn_version = core.cudnn_version()
        if cudnn_version < 0:
            return None
        else:
            return cudnn_version
    else:
        return _cudnn_version
