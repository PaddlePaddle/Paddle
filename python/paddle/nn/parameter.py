# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.base.framework import EagerParamBase
from paddle.tensor.creation import to_tensor


class Parameter(EagerParamBase):
    """
    Parameter is a subclass of EagerParamBase, which is a persistable Tensor
    that can be updated by optimizers during training.

    Args:
        data (Tensor, optional): The initial data for the Parameter.
            If None, an empty Tensor will be created. Default: None.
        requires_grad (bool): Whether this Parameter requires gradient computation.
            If True, the Parameter will accumulate gradients during backward pass.
            Default: True.

    Examples:
        >>> # Create a Parameter from existing Tensor
        >>> weight = paddle.to_tensor([1.0, 2.0, 3.0])
        >>> param = Parameter(weight)
        >>> print(param.requires_grad)  # True by default

        >>> # Create a Parameter without initial data
        >>> param = Parameter()
        >>> print(param.shape)  # empty tensor: []
    """

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = to_tensor([])
        param = EagerParamBase.from_tensor(data)
        param.stop_gradient = not requires_grad
        return param

    def __repr__(self):
        return f"Parameter containing:\n{super().__repr__()}"

    __str__ = __repr__
