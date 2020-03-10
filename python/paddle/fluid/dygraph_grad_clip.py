# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import copy
import six

import functools

from . import layers
from . import framework
from . import core
from .dygraph import base as imperative_base

__all__ = [
    'GradClipByValue',
    'GradClipByNorm',
    'GradClipByGlobalNorm',
]


class GradClipBase(object):
    def __str__(self):
        raise NotImplementedError()

    def _clip(self, para_and_grad):
        raise NotImplementedError

    @imperative_base.no_grad
    def __call__(self, para_and_grad):
        return self._clip(para_and_grad)


class GradClipByValue(GradClipBase):
    """
    Clips gradient values to the range [min_value, max_value].

    Given a gradient g, this operation clips its value to min_value and max_value.

    - Any values less than min_value are set to min_value.
    - Any values greater than max_value are set to max_value.

    Args:
        max_value (float): The maximum value to clip by. 
        min (float, optional): The minimum value to clip by. if not set by user, \
        will be set to -max_value(max_value MUST be positive) by framework. 

    Examples:
        .. code-block:: python
        
            import numpy as np
            import paddle
            import paddle.fluid as fluid

            from paddle.fluid.dygraph.base import to_variable
            from paddle.fluid.dygraph.nn import Linear

            from paddle.fluid.clip import GradClipByValue, GradClipByNorm, GradClipByGlobalNorm

            from paddle.fluid.optimizer import SGDOptimizer

            with fluid.dygraph.guard():
                value_clip = GradClipByValue( -1.0, 1.0 )
                sgd = SGDOptimizer(learning_rate=1.0)
                
                init_value = np.random.uniform( -1, 1, (10, 10)).astype('float32')

                linear = Linear( 10, 10)

                out = linear( to_variable(init_value) )

                loss = fluid.layers.reduce_mean( out )

                loss.backward()
                sgd.minimize(loss, grad_clip = value_clip)
            
    """

    @imperative_base.no_grad
    def __init__(self, min_value, max_value=None):

        if min_value is None:
            assert (max_value > 0.0)
            min_value = -max_value
        else:
            min_value = float(min_value)
        self.max_value = max_value
        self.min_value = min_value

    def __str__(self):
        return "ClipByValue, min = %f, max=%f" % (self.min_value,
                                                  self.max_value)

    def _clip(self, para_and_grad):
        out = []
        for p, g in para_and_grad:
            if g is None:
                out.append((p, g))
                continue

            new_grad = layers.clip(x=g, min=self.min_value, max=self.max_value)

            out.append((p, new_grad))

        return out


class GradClipByNorm(GradClipBase):
    """
    Clips tensor values to a maximum L2-norm.

    This operator limits the L2 norm of the input :math:`X` within :math:`max\_norm`.
    If the L2 norm of :math:`X` is less than or equal to :math:`max\_norm`, :math:`Out`
    will be the same as :math:`X`. If the L2 norm of :math:`X` is greater than
    :math:`max\_norm`, :math:`X` will be linearly scaled to make the L2 norm of
    :math:`Out` equal to :math:`max\_norm`, as shown in the following formula:

    .. math::

        Out = \\frac{max\_norm * X}{norm(X)},

    where :math:`norm(X)` represents the L2 norm of :math:`X`.

    Args:
        clip_norm (float): The maximum norm value

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.fluid as fluid

            from paddle.fluid.dygraph.base import to_variable
            from paddle.fluid.dygraph.nn import Linear

            from paddle.fluid.clip import GradClipByValue, GradClipByNorm, GradClipByGlobalNorm

            from paddle.fluid.optimizer import SGDOptimizer

            with fluid.dygraph.guard():
                norm_clip = GradClipByNorm( 5.0 )
                sgd = SGDOptimizer(learning_rate=1.0)
                
                init_value = np.random.uniform( -1, 1, (10, 10)).astype('float32')

                linear = Linear( 10, 10)

                out = linear( to_variable(init_value) )

                loss = fluid.layers.reduce_mean( out )

                loss.backward()
                sgd.minimize(loss, grad_clip = norm_clip)

    """

    @imperative_base.no_grad
    def __init__(self, clip_norm):
        self.clip_norm = clip_norm

    def __str__(self):
        return "ClipByNorm, clip_norm=%f" % self.clip_norm

    def _clip(self, para_and_grad):
        out = []

        for p, g in para_and_grad:
            if g is None:
                out.append((p, g))
                continue
            new_g = layers.clip_by_norm(x=g, max_norm=self.clip_norm)

            out.append((p, new_g))

        return out


class GradClipByGlobalNorm(GradClipBase):
    """
    Clips values of multiple tensors by the ratio of the sum of their norms.

    Given a list of tensors t_list, and a clipping ratio max_global_norm, this
    operation returns a list of clipped tensors list_clipped.

    To perform the clipping, the values :math:`t\_list[i]` are set to:

    .. math::

        t\_list[i] = t\_list[i] * \\frac{max\_global\_norm}{\max(global\_norm, max\_global\_norm)}

    where:

    .. math::

        global\_norm = \sqrt{\sum_{i=0}^{N-1}(l2norm(t\_list[i]))^2}

    If :math:`max\_global\_norm > global\_norm` then the entries in t_list remain as they are,
    otherwise they're all shrunk by the global ratio.

    Args:
        max_global_norm (float): The maximum norm value.
        dtype (str, optional): The type of max_global_norm. Default: "float32".

    Examples:
        .. code-block:: python
        
            import numpy as np
            import paddle
            import paddle.fluid as fluid

            from paddle.fluid.dygraph.base import to_variable
            from paddle.fluid.dygraph.nn import Linear

            from paddle.fluid.dygraph_grad_clip import GradClipByValue, GradClipByNorm, GradClipByGlobalNorm

            from paddle.fluid.optimizer import SGDOptimizer

            with fluid.dygraph.guard():
                gloabl_norm_clip = GradClipByGlobalNorm( 5.0 )
                sgd = SGDOptimizer(learning_rate=1.0)
                
                init_value = np.random.uniform( -1, 1, (10, 10)).astype('float32')

                linear = Linear( 10, 10)

                out = linear( to_variable(init_value) )

                loss = fluid.layers.reduce_mean( out )

                loss.backward()
                sgd.minimize(loss, grad_clip = gloabl_norm_clip)
   

    """

    @imperative_base.no_grad
    def __init__(self, max_global_norm, dtype='float32'):
        self.max_global_norm = layers.fill_constant(
            shape=[1], dtype=dtype, value=max_global_norm)

    def __str__(self):
        return "ClipByGlobalNorm, max_global_norm=%f" % (self.max_global_norm)

    def _clip(self, para_and_grad):

        out = []

        norm_arr = []
        for p, g in para_and_grad:
            if g is None:
                continue
            merge_grad = g
            if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                merge_grad = layers.merge_selected_rows(g)
                merge_grad = layers.get_tensor_from_selected_rows(merge_grad)
            power = layers.square(merge_grad)
            sum_t = layers.reduce_sum(power)
            norm_arr.append(sum_t)

        norm_global = layers.concat(norm_arr)
        norm_global = layers.reduce_sum(norm_global)
        norm_global = layers.sqrt(norm_global)

        clip_scale = self.max_global_norm / (layers.elementwise_max(
            x=norm_global, y=self.max_global_norm))

        for p, g in para_and_grad:
            if g is None:
                out.append((p, g))
                continue

            new_grad = g * clip_scale

            out.append((p, new_grad))

        return out
