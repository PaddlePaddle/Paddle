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

from __future__ import annotations

from typing import TYPE_CHECKING

from paddle.base.framework import EagerParamBase
from paddle.tensor.creation import to_tensor

if TYPE_CHECKING:
    from paddle import Tensor


class Parameter(EagerParamBase):
    """
    Parameter is a subclass of Tensor, which is a persistable Tensor
    that can be updated by optimizers during training.

    Args:
        data (Tensor, optional): The initial data for the Parameter.
            If None, an empty Tensor will be created. Default: None.
        requires_grad (bool, optional): Whether this Parameter requires gradient computation.
            If True, the Parameter will accumulate gradients during backward pass.
            Default: True.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> # Create a Parameter from existing Tensor
            >>> weight = paddle.to_tensor([1.0, 2.0, 3.0])
            >>> param = paddle.nn.Parameter(weight)
            >>> print(param)

            >>> # Create a Parameter without initial data
            >>> param = paddle.nn.Parameter()
            >>> print(param)
    """

    def __init__(
        self, data: Tensor | None = None, requires_grad: bool = True
    ) -> Parameter:
        if data is None:
            data = to_tensor([])
        super().__init__(data.shape, data.dtype, trainable=requires_grad)
        super()._set_impl(data)
        self._is_param = True

    def __repr__(self) -> str:
        return super().__repr__()

    __str__ = __repr__
