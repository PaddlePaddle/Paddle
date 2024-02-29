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
from paddle.base import core

__all__ = []


def with_metaclass(meta, *bases):
    class impl(meta):
        def __new__(cls, name, temp_bases, attrs):
            return meta(name, bases, attrs)

    return type.__new__(impl, "impl", (), {})


class PyLayerContext:
    """
    ``PyLayerContext`` can assist the :ref:`api_paddle_autograd_PyLayer` in implementing certain functionalities.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.autograd import PyLayer

            >>> class cus_tanh(PyLayer):
            ...     @staticmethod
            ...     def forward(ctx, x):
            ...         # ctx is a object of PyLayerContext.
            ...         y = paddle.tanh(x)
            ...         ctx.save_for_backward(y)
            ...         return y
            ...
            ...     @staticmethod
            ...     def backward(ctx, dy):
            ...         # ctx is a object of PyLayerContext.
            ...         y, = ctx.saved_tensor()
            ...         grad = dy * (1 - paddle.square(y))
            ...         return grad
    """

    def save_for_backward(self, *tensors):
        """
        Saves given tensors that backward need. Use ``saved_tensor`` in the `backward` to get the saved tensors.

        Note:
            This API should be called at most once, and only inside `forward`.

        Args:
            tensors(list of Tensors): Tensors to be stored.

        Returns:
            None

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.autograd import PyLayer

                >>> class cus_tanh(PyLayer):
                ...     @staticmethod
                ...     def forward(ctx, x):
                ...         # ctx is a context object that store some objects for backward.
                ...         y = paddle.tanh(x)
                ...         # Pass tensors to backward.
                ...         ctx.save_for_backward(y)
                ...         return y
                ...
                ...     @staticmethod
                ...     def backward(ctx, dy):
                ...         # Get the tensors passed by forward.
                ...         y, = ctx.saved_tensor()
                ...         grad = dy * (1 - paddle.square(y))
                ...         return grad

        """
        self.container = tensors

    def saved_tensor(self):
        """
        Get the tensors stored by ``save_for_backward``.

        Returns:
            list of Tensors or None: If context contains tensors stored by `save_for_backward`,
            then return these tensors, otherwise return None.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.autograd import PyLayer

                >>> class cus_tanh(PyLayer):
                ...     @staticmethod
                ...     def forward(ctx, x):
                ...         # ctx is a context object that store some objects for backward.
                ...         y = paddle.tanh(x)
                ...         # Pass tensors to backward.
                ...         ctx.save_for_backward(y)
                ...         return y
                ...
                ...     @staticmethod
                ...     def backward(ctx, dy):
                ...         # Get the tensors passed by forward.
                ...         y, = ctx.saved_tensor()
                ...         grad = dy * (1 - paddle.square(y))
                ...         return grad
        """
        return self.container

    def mark_not_inplace(self, *args):
        """
        Marks inputs as not inplace.
        This should be called at most once, only from inside the `forward` method,
        and all arguments should be Tensor inputs.

        If the Tensor returned by `forward` method is the same as the Tensor input of forward,
        and this Tensor is marked as not_inplace, then Paddle will help the user create a new Tensor as output.
        Thereby preventing the auto grad information of the input Tensor from being overwritten.

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> class Exp(paddle.autograd.PyLayer):
                ...     @staticmethod
                ...     def forward(ctx, x):
                ...         ctx.mark_not_inplace(x)
                ...         return x
                ...
                ...     @staticmethod
                ...     def backward(ctx, grad_output):
                ...         out = grad_output.exp()
                ...         return out

                >>> paddle.seed(2023)
                >>> x = paddle.randn((1, 1))
                >>> x.stop_gradient = False
                >>> attn_layers = []
                >>> for idx in range(0, 2):
                ...     attn_layers.append(Exp())

                >>> for step in range(0, 2):
                ...     a = x
                ...     for j in range(0,2):
                ...         a = attn_layers[j].apply(x)
                ...     a.backward()
        """
        self.not_inplace_tensors = args

    def mark_non_differentiable(self, *args):
        """
        Marks outputs as non-differentiable.
        This should be called at most once, only from inside the `forward` method,
        and all arguments should be tensor outputs.

        This will mark outputs as not requiring gradients, increasing the
        efficiency of backward computation. You still need to accept a gradient
        for each output in `backward`, but it's always going to
        be a zero tensor with the same shape as the shape of a corresponding
        output.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.autograd import PyLayer
                >>> import numpy as np

                >>> class Tanh(PyLayer):
                ...     @staticmethod
                ...     def forward(ctx, x):
                ...         a = x + x
                ...         b = x + x + x
                ...         ctx.mark_non_differentiable(a)
                ...         return a, b
                ...
                ...     @staticmethod
                ...     def backward(ctx, grad_a, grad_b):
                ...         assert np.equal(grad_a.numpy(), paddle.zeros([1]).numpy())
                ...         assert np.equal(grad_b.numpy(), paddle.ones([1], dtype="float64").numpy())
                ...         return grad_b

                >>> x = paddle.ones([1], dtype="float64")
                >>> x.stop_gradient = False
                >>> a, b = Tanh.apply(x)
                >>> b.sum().backward()
        """
        self.non_differentiable = args

    def set_materialize_grads(self, value: bool):
        """
        Sets whether to materialize output grad tensors. Default is True.

        This should be called only from inside the `forward` method.

        If True, undefined output grad tensors will be expanded to tensors full
        of zeros prior to calling the `backward` method.

        If False, undefined output grad tensors will be None.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.autograd import PyLayer
                >>> import numpy as np

                >>> class Tanh(PyLayer):
                ...     @staticmethod
                ...     def forward(ctx, x):
                ...         return x+x+x, x+x
                ...
                ...     @staticmethod
                ...     def backward(ctx, grad, grad2):
                ...         assert np.equal(grad2.numpy(), paddle.zeros([1]).numpy())
                ...         return grad

                >>> class Tanh2(PyLayer):
                ...     @staticmethod
                ...     def forward(ctx, x):
                ...         ctx.set_materialize_grads(False)
                ...         return x+x+x, x+x
                ...
                ...     @staticmethod
                ...     def backward(ctx, grad, grad2):
                ...         assert grad2==None
                ...         return grad

                >>> x = paddle.ones([1], dtype="float64")
                >>> x.stop_gradient = False
                >>> Tanh.apply(x)[0].backward()

                >>> x2 = paddle.ones([1], dtype="float64")
                >>> x2.stop_gradient = False
                >>> Tanh2.apply(x2)[0].backward()
        """
        self.materialize_grads = value


class PyLayerBackward(core.eager.PyLayer, PyLayerContext):
    def backward(self, *args):
        return self._forward_cls.backward(self, *args)


class PyLayerMeta(type):
    def __init__(cls, name, bases, attrs):
        cls._backward_function = type(
            name + '_backward', (PyLayerBackward,), {"_forward_cls": cls}
        )

        return super().__init__(name, bases, attrs)


class PyLayer(with_metaclass(PyLayerMeta, core.eager.PyLayer, PyLayerContext)):
    """
    Paddle implements Python custom operators on the PaddlePaddle framework by creating a subclass of
    ``PyLayer``, which must comply with the following rules:

    1. The subclass must contain static ``forward`` and ``backward`` functions, with the first argument being
    :ref:`api_paddle_autograd_PyLayerContext`. If a returned value in ``backward`` corresponds to a ``Tensor`` that
    requires gradients in ``forward``, the returned value must be a ``Tensor``.

    2. Except for the first argument, other arguments of ``backward`` are gradients of the output ``Tensors``
    of ``forward``. Therefore, the number of input ``Tensor`` in ``backward`` must be the same as the number
    of output ``Tensor`` in ``forward``. If you need to use input ``Tensor`` from ``forward`` in ``backward``,
    you can save these ``Tensors`` by inputting them into :ref:`api_paddle_autograd_PyLayerContext`'s
    ``save_for_backward`` method and use them in ``backward`` later.

    3. The output of ``backward`` can be ``Tensor`` or ``list/tuple(Tensor)``, which are gradients of the
    output ``Tensor`` of ``forward``. Therefore, the number of output ``Tensor`` in ``backward`` is the same
    as the number of input ``Tensor`` in ``forward``.

    After building the custom operator, apply it by running the ``apply`` method.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.autograd import PyLayer

            >>> class cus_tanh(PyLayer):
            ...     @staticmethod
            ...     def forward(ctx, x):
            ...         y = paddle.tanh(x)
            ...         # Pass tensors to backward.
            ...         ctx.save_for_backward(y)
            ...         return y
            ...
            ...     @staticmethod
            ...     def backward(ctx, dy):
            ...         # Get the tensors passed by forward.
            ...         y, = ctx.saved_tensor()
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

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        It is to be overloaded by subclasses. It must accept a object of :ref:`api_paddle_autograd_PyLayerContext` as
        the first argument, followed by any number of arguments (tensors or other types).
        `None` can not be included in the returned result.

        Args:
            *args(tuple): input of PyLayer.
            **kwargs(dict): input of PyLayer.

        Returns:
            tensors or other types : output of PyLayer.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.autograd import PyLayer

                >>> class cus_tanh(PyLayer):
                ...     @staticmethod
                ...     def forward(ctx, x):
                ...         y = paddle.tanh(x)
                ...         # Pass tensors to backward.
                ...         ctx.save_for_backward(y)
                ...         return y
                ...
                ...     @staticmethod
                ...     def backward(ctx, dy):
                ...         # Get the tensors passed by forward.
                ...         y, = ctx.saved_tensor()
                ...         grad = dy * (1 - paddle.square(y))
                ...         return grad
        """
        raise NotImplementedError(
            "You must implement the forward function for PyLayer."
        )

    @staticmethod
    def backward(ctx, *args):
        """
        This is a function to calculate the gradient. It is to be overloaded by subclasses.
        It must accept a object of :ref:`api_paddle_autograd_PyLayerContext` as the first
        argument, and the rest arguments are the gradient of forward's output tensors.
        Output tensors of backward are the gradient of forward's input tensors.

        Args:
            *args(tuple): The gradient of forward's output tensor(s).
            **kwargs(dict): The gradient of forward's output tensor(s).

        Returns:
            Tensor or list of Tensors: The gradient of forward's input tensor(s).

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.autograd import PyLayer

                >>> class cus_tanh(PyLayer):
                ...     @staticmethod
                ...     def forward(ctx, x):
                ...         y = paddle.tanh(x)
                ...         # Pass tensors to backward.
                ...         ctx.save_for_backward(y)
                ...         return y
                ...
                ...     @staticmethod
                ...     def backward(ctx, dy):
                ...         # Get the tensors passed by forward.
                ...         y, = ctx.saved_tensor()
                ...         grad = dy * (1 - paddle.square(y))
                ...         return grad
        """

        raise NotImplementedError(
            "You must implement the backward function for PyLayer."
        )


def once_differentiable(backward):
    def wrapper(ctx, *args):
        with paddle.base.dygraph.no_grad():
            outputs = backward(ctx, *args)
        return outputs

    return wrapper
