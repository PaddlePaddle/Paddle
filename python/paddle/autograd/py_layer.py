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
from paddle.fluid.framework import dygraph_only
from paddle.fluid.dygraph.amp.auto_cast import amp_state
from paddle.amp.auto_cast import auto_cast
from paddle.fluid import core

__all__ = []


class LegacyPyLayerContext(object):
    """
    The object of this class is a context that is used in PyLayer to enhance the function.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.autograd import PyLayer

            class cus_tanh(PyLayer):
                @staticmethod
                def forward(ctx, x):
                    # ctx is a object of PyLayerContext.
                    y = paddle.tanh(x)
                    ctx.save_for_backward(y)
                    return y

                @staticmethod
                def backward(ctx, dy):
                    # ctx is a object of PyLayerContext.
                    y, = ctx.saved_tensor()
                    grad = dy * (1 - paddle.square(y))
                    return grad
    """

    def __init__(self):
        self.container = None
        self._amp_state = amp_state()

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

                import paddle
                from paddle.autograd import PyLayer

                class cus_tanh(PyLayer):
                    @staticmethod
                    def forward(ctx, x):
                        # ctx is a context object that store some objects for backward.
                        y = paddle.tanh(x)
                        # Pass tensors to backward.
                        ctx.save_for_backward(y)
                        return y

                    @staticmethod
                    def backward(ctx, dy):
                        # Get the tensors passed by forward.
                        y, = ctx.saved_tensor()
                        grad = dy * (1 - paddle.square(y))
                        return grad

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

                import paddle
                from paddle.autograd import PyLayer

                class cus_tanh(PyLayer):
                    @staticmethod
                    def forward(ctx, x):
                        # ctx is a context object that store some objects for backward.
                        y = paddle.tanh(x)
                        # Pass tensors to backward.
                        ctx.save_for_backward(y)
                        return y

                    @staticmethod
                    def backward(ctx, dy):
                        # Get the tensors passed by forward.
                        y, = ctx.saved_tensor()
                        grad = dy * (1 - paddle.square(y))
                        return grad
        """

        return self.container


def with_mateclass(meta, *bases):

    class impl(meta):

        def __new__(cls, name, temp_bases, attrs):
            return meta(name, bases, attrs)

    return type.__new__(impl, "impl", (), {})


class CPyLayer(object):

    @classmethod
    @dygraph_only
    def apply(cls, *args, **kwargs):
        """
        After building the custom PyLayer, run it through the ``apply``.

        Args:
            *args(tuple): input of PyLayer.
            **kwargs(dict): input of PyLayer.

        Returns:
            tensors or other types : output of PyLayer.

        Examples:
            .. code-block:: python

                import paddle
                from paddle.autograd import PyLayer

                class cus_tanh(PyLayer):
                    @staticmethod
                    def forward(ctx, x, func1, func2=paddle.square):
                        ctx.func = func2
                        y = func1(x)
                        # Pass tensors to backward.
                        ctx.save_for_backward(y)
                        return y

                    @staticmethod
                    def backward(ctx, dy):
                        # Get the tensors passed by forward.
                        y, = ctx.saved_tensor()
                        grad = dy * (1 - ctx.func(y))
                        return grad


                data = paddle.randn([2, 3], dtype="float64")
                data.stop_gradient = False
                # run custom Layer.
                z = cus_tanh.apply(data, func1=paddle.tanh)
        """
        place = paddle.fluid.framework._current_expected_place()
        with paddle.fluid.dygraph.no_grad():
            return core.pylayer_apply(place, cls, *args, **kwargs)


class PyLayerBackward(LegacyPyLayerContext):

    def backward(self, *args, **kwargs):
        with paddle.fluid.dygraph.guard():
            with paddle.fluid.dygraph.no_grad():
                if self._amp_state and 'enable' in self._amp_state and self._amp_state[
                        'enable']:
                    with auto_cast(**args[0]._amp_state):
                        return self._forward_cls.backward(*args, **kwargs)
                else:

                    return self._forward_cls.backward(*args, **kwargs)
                return self._forward_cls.backward(*args, **kwargs)


class LayerMeta(type):

    def __init__(cls, name, bases, attrs):
        cls._backward_function = type(name + '_backward', (PyLayerBackward, ),
                                      {"_forward_cls": cls})

        return super(LayerMeta, cls).__init__(name, bases, attrs)


class LegacyPyLayer(with_mateclass(LayerMeta, CPyLayer)):
    """
    Build a custom `Layer` by creating subclasses. Subclasses need to follow the following rules:
    1. Subclasses contain `forward` and `backward` function. Both forward and backward are @staticmethod.
    Their first argument should be a context and `None` can not be included in the returned result.
    2. Input of backward contains a context as the first argument, and the rest arguments are the
    gradient of forward's output tensors. so the number of backward's input tensors equal to
    the number of forward output tensors. If you need the forward's inputs or outputs in `backward`,
    you can use `save_for_backward` to store the required tensors, and then use them in the backward.
    3. Output of backward function can only be `Tensor` or tuple/list of `Tensor`.
    Output tensors of backward are the gradient of forward's input tensors,
    so the number of backward's output tensors equal to the number of forward input tensors.
    After building the custom Layer, run it through the `apply` method.


    Examples:
        .. code-block:: python

            import paddle
            from paddle.autograd import PyLayer

            # Inherit from PyLayer
            class cus_tanh(PyLayer):
                @staticmethod
                def forward(ctx, x, func1, func2=paddle.square):
                    # ctx is a context object that store some objects for backward.
                    ctx.func = func2
                    y = func1(x)
                    # Pass tensors to backward.
                    ctx.save_for_backward(y)
                    return y

                @staticmethod
                # forward has only one output, so there is only one gradient in the input of backward.
                def backward(ctx, dy):
                    # Get the tensors passed by forward.
                    y, = ctx.saved_tensor()
                    grad = dy * (1 - ctx.func(y))
                    # forward has only one input, so only one gradient tensor is returned.
                    return grad


            data = paddle.randn([2, 3], dtype="float64")
            data.stop_gradient = False
            z = cus_tanh.apply(data, func1=paddle.tanh)
            z.mean().backward()

            print(data.grad)

    """

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        It is to be overloaded by subclasses. It must accept a object of `PyLayerContext` as
        the first argument, followed by any number of arguments (tensors or other types).
        `None` can not be included in the returned result.

        Args:
            *args(tuple): input of PyLayer.
            **kwargs(dict): input of PyLayer.

        Returns:
            tensors or other types : output of PyLayer.

        Examples:
            .. code-block:: python

                import paddle
                from paddle.autograd import PyLayer

                class cus_tanh(PyLayer):
                    @staticmethod
                    def forward(ctx, x):
                        y = paddle.tanh(x)
                        # Pass tensors to backward.
                        ctx.save_for_backward(y)
                        return y

                    @staticmethod
                    def backward(ctx, dy):
                        # Get the tensors passed by forward.
                        y, = ctx.saved_tensor()
                        grad = dy * (1 - paddle.square(y))
                        return grad
        """
        raise NotImplementedError(
            "You must implement the forward function for PyLayer.")

    @staticmethod
    def backward(ctx, *args, **kwargs):
        """
        This is a function to calculate the gradient. It is to be overloaded by subclasses.
        It must accept a object of `PyLayerContext` as the first argument, and the rest
        arguments are the gradient of forward's output tensors. Output tensors of backward
        are the gradient of forward's input tensors.

        Args:
            *args(tuple): The gradient of forward's output tensor(s).
            **kwargs(dict): The gradient of forward's output tensor(s).

        Returns:
            Tensor or list of Tensors: The gradient of forward's input tensor(s).

        Examples:
            .. code-block:: python

                import paddle
                from paddle.autograd import PyLayer

                class cus_tanh(PyLayer):
                    @staticmethod
                    def forward(ctx, x):
                        y = paddle.tanh(x)
                        # Pass tensors to backward.
                        ctx.save_for_backward(y)
                        return y

                    @staticmethod
                    def backward(ctx, dy):
                        # Get the tensors passed by forward.
                        y, = ctx.saved_tensor()
                        grad = dy * (1 - paddle.square(y))
                        return grad
        """

        raise NotImplementedError(
            "You must implement the backward function for PyLayer.")


class EagerPyLayerContext(object):

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

                import paddle
                from paddle.autograd import PyLayer

                class cus_tanh(PyLayer):
                    @staticmethod
                    def forward(ctx, x):
                        # ctx is a context object that store some objects for backward.
                        y = paddle.tanh(x)
                        # Pass tensors to backward.
                        ctx.save_for_backward(y)
                        return y

                    @staticmethod
                    def backward(ctx, dy):
                        # Get the tensors passed by forward.
                        y, = ctx.saved_tensor()
                        grad = dy * (1 - paddle.square(y))
                        return grad

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

                import paddle
                from paddle.autograd import PyLayer

                class cus_tanh(PyLayer):
                    @staticmethod
                    def forward(ctx, x):
                        # ctx is a context object that store some objects for backward.
                        y = paddle.tanh(x)
                        # Pass tensors to backward.
                        ctx.save_for_backward(y)
                        return y

                    @staticmethod
                    def backward(ctx, dy):
                        # Get the tensors passed by forward.
                        y, = ctx.saved_tensor()
                        grad = dy * (1 - paddle.square(y))
                        return grad
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

                import paddle

                class Exp(paddle.autograd.PyLayer):
                    @staticmethod
                    def forward(ctx, x):
                        ctx.mark_not_inplace(x)
                        return x

                    @staticmethod
                    def backward(ctx, grad_output):
                        out = grad_output.exp()
                        return out

                x = paddle.randn((1, 1))
                x.stop_gradient = False
                attn_layers = []
                for idx in range(0, 2):
                    attn_layers.append(Exp())

                for step in range(0, 2):
                    a = x
                    for j in range(0,2):
                        a = attn_layers[j].apply(x)
                    a.backward()
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

                import os
                os.environ['FLAGS_enable_eager_mode'] = '1'
                import paddle
                from paddle.autograd import PyLayer
                import numpy as np

                class Tanh(PyLayer):
                    @staticmethod
                    def forward(ctx, x):
                        a = x + x
                        b = x + x + x
                        ctx.mark_non_differentiable(a)
                        return a, b

                    @staticmethod
                    def backward(ctx, grad_a, grad_b):
                        assert np.equal(grad_a.numpy(), paddle.zeros([1]).numpy())
                        assert np.equal(grad_b.numpy(), paddle.ones([1], dtype="float64").numpy())
                        return grad_b

                x = paddle.ones([1], dtype="float64")
                x.stop_gradient = False
                a, b = Tanh.apply(x)
                b.sum().backward()
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

                import os
                os.environ['FLAGS_enable_eager_mode'] = '1'
                import paddle
                from paddle.autograd import PyLayer
                import numpy as np

                class Tanh(PyLayer):
                    @staticmethod
                    def forward(ctx, x):
                        return x+x+x, x+x

                    @staticmethod
                    def backward(ctx, grad, grad2):
                        assert np.equal(grad2.numpy(), paddle.zeros([1]).numpy())
                        return grad

                class Tanh2(PyLayer):
                    @staticmethod
                    def forward(ctx, x):
                        ctx.set_materialize_grads(False)
                        return x+x+x, x+x

                    @staticmethod
                    def backward(ctx, grad, grad2):
                        assert grad2==None
                        return grad

                x = paddle.ones([1], dtype="float64")
                x.stop_gradient = False
                Tanh.apply(x)[0].backward()

                x2 = paddle.ones([1], dtype="float64")
                x2.stop_gradient = False
                Tanh2.apply(x2)[0].backward()
        """
        self.materialize_grads = value


class EagerPyLayerBackward(core.eager.PyLayer, EagerPyLayerContext):

    def backward(self, *args):
        return self._forward_cls.backward(self, *args)


class EagerPyLayerMeta(type):

    def __init__(cls, name, bases, attrs):
        cls._backward_function = type(name + '_backward',
                                      (EagerPyLayerBackward, ),
                                      {"_forward_cls": cls})

        return super(EagerPyLayerMeta, cls).__init__(name, bases, attrs)


class EagerPyLayer(
        with_mateclass(EagerPyLayerMeta, core.eager.PyLayer,
                       EagerPyLayerContext)):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        It is to be overloaded by subclasses. It must accept a object of `PyLayerContext` as
        the first argument, followed by any number of arguments (tensors or other types).
        `None` can not be included in the returned result.

        Args:
            *args(tuple): input of PyLayer.
            **kwargs(dict): input of PyLayer.

        Returns:
            tensors or other types : output of PyLayer.

        Examples:
            .. code-block:: python

                import paddle
                from paddle.autograd import PyLayer

                class cus_tanh(PyLayer):
                    @staticmethod
                    def forward(ctx, x):
                        y = paddle.tanh(x)
                        # Pass tensors to backward.
                        ctx.save_for_backward(y)
                        return y

                    @staticmethod
                    def backward(ctx, dy):
                        # Get the tensors passed by forward.
                        y, = ctx.saved_tensor()
                        grad = dy * (1 - paddle.square(y))
                        return grad
        """
        raise NotImplementedError(
            "You must implement the forward function for PyLayer.")

    @staticmethod
    def backward(ctx, *args):
        """
        This is a function to calculate the gradient. It is to be overloaded by subclasses.
        It must accept a object of `PyLayerContext` as the first argument, and the rest
        arguments are the gradient of forward's output tensors. Output tensors of backward
        are the gradient of forward's input tensors.

        Args:
            *args(tuple): The gradient of forward's output tensor(s).
            **kwargs(dict): The gradient of forward's output tensor(s).

        Returns:
            Tensor or list of Tensors: The gradient of forward's input tensor(s).

        Examples:
            .. code-block:: python

                import paddle
                from paddle.autograd import PyLayer

                class cus_tanh(PyLayer):
                    @staticmethod
                    def forward(ctx, x):
                        y = paddle.tanh(x)
                        # Pass tensors to backward.
                        ctx.save_for_backward(y)
                        return y

                    @staticmethod
                    def backward(ctx, dy):
                        # Get the tensors passed by forward.
                        y, = ctx.saved_tensor()
                        grad = dy * (1 - paddle.square(y))
                        return grad
        """

        raise NotImplementedError(
            "You must implement the backward function for PyLayer.")


def once_differentiable(backward):

    def wrapper(ctx, *args):
        with paddle.fluid.dygraph.no_grad():
            outputs = backward(ctx, *args)
        return outputs

    return wrapper
