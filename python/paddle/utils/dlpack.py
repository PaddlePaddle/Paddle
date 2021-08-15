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

import paddle


def to_dlpack(x):
    """Returns the DLPack capsule representing the corresponding tensor.

    Args:
        x (Tensor): a Paddle tensor to be converted to DLPack capsule.

    Returns:
        A PyCapsule object with the dltensor, which shares the memory to other frameworks. 
        The PyCapsule can only be consumed once.
    """
    return x.value().get_tensor()._to_dlpack()


def from_dlpack(dlpack):
    """Decodes a DLPack to a tensor.
    
    Args:
        dlpack (PyCapsule): a PyCapsule object with the dltensor.

    Returns:
        out (Tensor): a tensor decoded from DLPack.
    """
    out = paddle.fluid.core.from_dlpack(dlpack)
    out = paddle.to_tensor(out)
    return out
