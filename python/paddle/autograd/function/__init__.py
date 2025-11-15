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

"""
This module provides PyTorch-compatible API for custom autograd Functions.
It is an alias/compatibility layer over paddle.autograd.PyLayer.
"""

from paddle.autograd.py_layer import (
    PyLayer as _PyLayer,
    PyLayerContext as _PyLayerContext,
)

__all__ = [
    'Function',
    'FunctionCtx',
]


class FunctionCtx(_PyLayerContext):
    """
    PyTorch-compatible context object for custom autograd Functions.
    This is an alias/wrapper for :ref:`api_paddle_autograd_PyLayerContext`.

    The main difference is that ``saved_tensors`` is provided as a property
    (matching PyTorch's API) instead of a method call.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.autograd.function import Function

            >>> class cus_tanh(Function):
            ...     @staticmethod
            ...     def forward(ctx, x):
            ...         y = paddle.tanh(x)
            ...         ctx.save_for_backward(x, y)
            ...         return y
            ...
            ...     @staticmethod
            ...     def backward(ctx, dy):
            ...         x, y = ctx.saved_tensors  # Use as property, PyTorch style
            ...         grad = dy * (1 - paddle.square(y))
            ...         return grad
    """

    @property
    def saved_tensors(self):
        """
        Get the tensors stored by ``save_for_backward``.
        This property provides PyTorch-compatible access (as an attribute rather than method call).

        Returns:
            tuple of Tensors: The tensors saved during forward pass.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.autograd.function import Function

                >>> class cus_tanh(Function):
                ...     @staticmethod
                ...     def forward(ctx, x):
                ...         y = paddle.tanh(x)
                ...         ctx.save_for_backward(x, y)
                ...         return y
                ...
                ...     @staticmethod
                ...     def backward(ctx, dy):
                ...         # Access saved tensors as property (PyTorch style)
                ...         x, y = ctx.saved_tensors
                ...         grad = dy * (1 - paddle.square(y))
                ...         return grad
        """
        return self.saved_tensor()


# Create a custom metaclass to inject FunctionCtx instead of PyLayerContext
class FunctionMeta(type):
    def __init__(cls, name, bases, attrs):
        # Create backward function with FunctionCtx
        from paddle.base import core

        class FunctionBackward(core.eager.PyLayer, FunctionCtx):
            def backward(self, *args):
                return self._forward_cls.backward(self, *args)

        cls._backward_function = type(
            name + '_backward', (FunctionBackward,), {"_forward_cls": cls}
        )

        super().__init__(name, bases, attrs)


class Function(_PyLayer, metaclass=FunctionMeta):
    """
    PyTorch-compatible custom autograd Function.
    This is an alias/compatibility layer for :ref:`api_paddle_autograd_PyLayer`.

    Create custom autograd Functions by subclassing ``Function`` and implementing
    the ``forward`` and ``backward`` static methods. This API is designed to be
    compatible with ``torch.autograd.Function``.

    Rules:
    1. The subclass must contain static ``forward`` and ``backward`` methods,
       with the first argument being ``FunctionCtx`` (or ``ctx``).

    2. Arguments of ``backward`` (except ctx) are gradients of the outputs of ``forward``.

    3. Outputs of ``backward`` are gradients of the inputs of ``forward``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.autograd.function import Function

            >>> class cus_tanh(Function):
            ...     @staticmethod
            ...     def forward(ctx, x):
            ...         y = paddle.tanh(x)
            ...         ctx.save_for_backward(y)
            ...         return y
            ...
            ...     @staticmethod
            ...     def backward(ctx, dy):
            ...         y, = ctx.saved_tensors  # PyTorch-compatible property access
            ...         grad = dy * (1 - paddle.square(y))
            ...         return grad

            >>> paddle.seed(2023)
            >>> data = paddle.randn([2, 3], dtype="float64")
            >>> data.stop_gradient = False
            >>> z = cus_tanh.apply(data)
            >>> z.mean().backward()
            >>> print(data.grad)
            Tensor(shape=[2, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[0.16604150, 0.05858341, 0.14051214],
             [0.15677770, 0.01564609, 0.02991660]])
    """

    pass
