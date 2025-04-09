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
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from paddle.base import core

if TYPE_CHECKING:
    from collections.abc import Callable

    from paddle import Tensor
__all__ = []


class saved_tensors_hooks:
    """
    Dynamic graph, registers a pair of pack / unpack hooks for saved tensors.

    Parameters:
        pack_hook (function): The pack hook will be called every time the forward
            operation inputs/outputs tensors need be saved for backward. Then you
            can save it to CPU or Disk. The input of `pack_hook` is a tensor need
            be saved. The output of `pack_hook` is then stored information instead
            of the original tensor. `pack_hook` will also be called while any
            tensor need be saved by `PyLayerContext.save_for_backward`. If a tensor
            saved for backward is no need buffer, `pack_hook` will not be called.
            Only the tensor saved for backward is DenseTensor, `pack_hook` will be
            called.
        unpack_hook (function): The unpack hook will be called every time the
            backward need use the saved inputs/outputs tensors. Then you can reload
            the tensor and return it to paddle framework. The input of `unpack_hook`
            is the information returned by `pack_hook`. The output of `unpack_hook`
            is a tensor reloaded by the information, and the tensor must has the same
            content as the original tensor passed as input to the corresponding
            `pack_hook`.

    Returns:
            None

    Examples:
        .. code-block:: python
            :name: code-example1

            >>> # Example1
            >>> import paddle

            >>> def pack_hook(x):
            ...     print("Packing", x)
            ...     return x.numpy()

            >>> def unpack_hook(x):
            ...     print("UnPacking", x)
            ...     return paddle.to_tensor(x)

            >>> a = paddle.ones([3,3])
            >>> b = paddle.ones([3,3]) * 2
            >>> a.stop_gradient = False
            >>> b.stop_gradient = False
            >>> with paddle.autograd.saved_tensors_hooks(pack_hook, unpack_hook):
            ...     y = paddle.multiply(a, b)
            >>> y.sum().backward()

        .. code-block:: python
            :name: code-example2

            >>> # Example2
            >>> import paddle
            >>> from paddle.autograd import PyLayer

            >>> class cus_multiply(PyLayer):
            ...     @staticmethod
            ...     def forward(ctx, a, b):
            ...         y = paddle.multiply(a, b)
            ...         ctx.save_for_backward(a, b)
            ...         return y
            ...
            ...     @staticmethod
            ...     def backward(ctx, dy):
            ...         a,b = ctx.saved_tensor()
            ...         grad_a = dy * a
            ...         grad_b = dy * b
            ...         return grad_a, grad_b

            >>> def pack_hook(x):
            ...     print("Packing", x)
            ...     return x.numpy()

            >>> def unpack_hook(x):
            ...     print("UnPacking", x)
            ...     return paddle.to_tensor(x)

            >>> a = paddle.ones([3,3])
            >>> b = paddle.ones([3,3]) * 2
            >>> a.stop_gradient = False
            >>> b.stop_gradient = False
            >>> with paddle.autograd.saved_tensors_hooks(pack_hook, unpack_hook):
            ...     y = cus_multiply.apply(a, b)
            >>> y.sum().backward()
    """

    def __init__(
        self,
        pack_hook: Callable[[Tensor], Any | None],
        unpack_hook: Callable[[Any], Tensor | None],
    ) -> None:
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook

    def __enter__(self) -> None:
        core.eager.register_saved_tensors_hooks(
            self.pack_hook, self.unpack_hook
        )

    def __exit__(self, *args: object) -> None:
        core.eager.reset_saved_tensors_hooks()
