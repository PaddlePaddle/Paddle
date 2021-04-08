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
from paddle.fluid import core
__all__ = ['PyLayer']


class PyLayerContext(object):
    def __init__(self):
        self.container = None

    def save_for_backward(self, *tensors):
        """
        Saves given tensors that backward need. Use `saved_tensor` in the `backward` to get the saved tensors.
        
        .. note::
            This API should be called at most once, and only inside `forward`. 

        Args:
            tensors(list of Tensors): Tensors to be stored.

        Returns:
            None

        """
        self.container = tensors

    def saved_tensor(self):
        """
        Get the tensors stored by `save_for_backward`.

        Returns:
            list of Tensors or None: If context contains tensors stored by `save_for_backward`, 
                then return these tensors, otherwise return None.
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
        place = paddle.fluid.framework._current_expected_place()
        with paddle.fluid.dygraph.no_grad():
            return core.PyLayer_apply(place, cls, *args, **kwargs)


class PyLayerBackward(PyLayerContext):
    def backward(self, *args, **kwargs):
        with paddle.fluid.dygraph.no_grad():
            return self._forward_cls.backward(*args, **kwargs)


class LayerMeta(type):
    def __init__(cls, name, bases, attrs):
        cls._backward_function = type(name + '_backward', (PyLayerBackward, ),
                                      {"_forward_cls": cls})

        return super(LayerMeta, cls).__init__(name, bases, attrs)


class PyLayer(with_mateclass(LayerMeta, CPyLayer)):
    """
    Build a custom `Layer` by creating subclasses. Subclasses need to follow the following rules:
        **1. Subclasses contain `forward` and `backward` function. Both forward and backward are @staticmethod.
            Their first argument should be a context and `None` can be included in the returned result.
        **2. Input of backward contains a context as the first argument, and the rest arguments are the 
            gradient of forward's output tensors. so the number of backward's input tensors equal to 
            the number of forward output tensors. If you need the forward's inputs or outputs in `backward`, 
            you can use `save_for_backward` to store the required tensors, and then use them in the backward.
        **3. Output of backward function can only be `Tensor` or tuple/list of `Tensor`.
            Output tensors of backward are the gradient of forward's input tensors, 
            so the number of backward's output tensors equal to the number of forward input tensors.
    
    Examples:
            .. code-block:: python

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
        raise NotImplementedError(
            "You must implement the forward function for custom layer.")

    @staticmethod
    def backward(ctx, *args, **kwargs):
        raise NotImplementedError(
            "You must implement the backward function for custom layer.")
