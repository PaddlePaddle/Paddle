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
from types import FunctionType
import copy
import six
import warnings

import functools

from . import layers
from . import framework
from . import core
from .dygraph import base as imperative_base
from .clip import _correct_clip_op_role_var

__all__ = [
    'GradClipBase', 'GradClipByValue', 'GradClipByNorm', 'GradClipByGlobalNorm',
    'ClipByValue', 'ClipByNorm', 'ClipByGlobalNorm'
]


class GradClipBase(object):
    def __init__(self, need_clip=None):
        if need_clip is not None and not isinstance(need_clip, FunctionType):
            raise TypeError(
                "The type of need_clip must be funciton, and it can filter "
                "out parameter that does't need gradient clip, please refer to "
                "documention of API:fluid.ClipByValue/fluid.ClipByGlobalNorm/"
                "fluid.ClipByNorm!")
        self._need_clip_func = need_clip

    def __str__(self):
        raise NotImplementedError()

    @imperative_base.no_grad
    def _dygraph_clip(self, params_grads):
        raise NotImplementedError

    def _static_clip(self, params_grads):
        raise NotImplementedError

    def __call__(self, params_grads):
        assert len(
            params_grads
        ) > 0, "The number of trainable parameters should be greater than 0."
        if getattr(params_grads[0][0], 'gradient_clip_attr', None) is not None:
            warnings.warn(
                "'set_gradient_clip' will be ineffective, because you have "
                "pass 'grad_clip' into 'minimize'. So, 'set_gradient_clip' "
                "is redundant and you can remove it.")
        if framework.in_dygraph_mode():
            return self._dygraph_clip(params_grads)
        else:
            return self._static_clip(params_grads)


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

    def __init__(self, min_value, max_value, need_clip=None):
        super(GradClipByValue, self).__init__(need_clip)
        if min_value is None:
            assert (max_value > 0.0)
            min_value = -max_value
        self.max_value = float(max_value)
        self.min_value = float(min_value)

    def __str__(self):
        return "Gradient Clip By Value, min = %f, max=%f" % (self.min_value,
                                                             self.max_value)

    @imperative_base.no_grad
    def _dygraph_clip(self, params_grads):
        params_and_grads = []
        for p, g in params_grads:
            if g is None:
                params_and_grads.append((p, g))
                continue
            if self._need_clip_func is not None and not self._need_clip_func(p):
                params_and_grads.append((p, g))
                continue
            new_grad = layers.clip(x=g, min=self.min_value, max=self.max_value)
            params_and_grads.append((p, new_grad))
        return params_and_grads

    def _static_clip(self, params_grads):
        params_and_grads = []
        with framework.name_scope('gradient_clip'):
            for p, g in params_grads:
                if g is None:
                    params_and_grads.append((p, g))
                    continue
                if self._need_clip_func is not None and not self._need_clip_func(
                        p):
                    params_and_grads.append((p, g))
                    continue

                with p.block.program._optimized_guard([p, g]):
                    new_grad = layers.clip(
                        x=g, min=self.min_value, max=self.max_value)
                params_and_grads.append((p, new_grad))
        _correct_clip_op_role_var(params_and_grads)
        return params_and_grads


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

    def __init__(self, clip_norm, need_clip=None):
        self.clip_norm = float(clip_norm)
        super(GradClipByNorm, self).__init__(need_clip)

    def __str__(self):
        return "Gradient Clip By Norm, clip_norm=%f" % self.clip_norm

    @imperative_base.no_grad
    def _dygraph_clip(self, params_grads):
        params_and_grads = []
        for p, g in params_grads:
            if g is None:
                params_and_grads.append((p, g))
                continue
            if self._need_clip_func is not None and not self._need_clip_func(p):
                params_and_grads.append((p, g))
                continue
            new_grad = layers.clip_by_norm(x=g, max_norm=self.clip_norm)
            params_and_grads.append((p, new_grad))
        return params_and_grads

    def _static_clip(self, params_grads):
        params_and_grads = []
        with framework.name_scope('gradient_clip'):
            for p, g in params_grads:
                if g is None:
                    params_and_grads.append((p, g))
                    continue
                if self._need_clip_func is not None and not self._need_clip_func(
                        p):
                    params_and_grads.append((p, g))
                    continue

                with p.block.program._optimized_guard([p, g]):
                    new_grad = layers.clip_by_norm(x=g, max_norm=self.clip_norm)
                params_and_grads.append((p, new_grad))
        _correct_clip_op_role_var(params_and_grads)
        return params_and_grads


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

    def __init__(self, clip_norm, need_clip=None):
        self.clip_norm = float(clip_norm)
        super(GradClipByGlobalNorm, self).__init__(need_clip)

    def __str__(self):
        return "Gradient Clip By GlobalNorm, global_norm=%f" % (self.clip_norm)

    @imperative_base.no_grad
    def _dygraph_clip(self, params_grads):
        params_and_grads = []
        sum_square_list = []
        for p, g in params_grads:
            if g is None:
                continue
            if self._need_clip_func is not None and not self._need_clip_func(p):
                continue
            merge_grad = g
            if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                merge_grad = layers.merge_selected_rows(g)
                merge_grad = layers.get_tensor_from_selected_rows(merge_grad)
            square = layers.square(merge_grad)
            sum_square = layers.reduce_sum(square)
            sum_square_list.append(sum_square)

        if len(sum_square_list) == 0:
            return params_grads

        global_norm_var = layers.concat(sum_square_list)
        global_norm_var = layers.reduce_sum(global_norm_var)
        global_norm_var = layers.sqrt(global_norm_var)
        max_global_norm = layers.fill_constant(
            shape=[1], dtype='float32', value=self.clip_norm)
        clip_var = max_global_norm / (layers.elementwise_max(
            x=global_norm_var, y=max_global_norm))
        for p, g in params_grads:
            if g is None:
                params_and_grads.append((p, g))
                continue
            if self._need_clip_func is not None and not self._need_clip_func(p):
                params_and_grads.append((p, g))
                continue
            new_grad = g * clip_var
            params_and_grads.append((p, new_grad))

        return params_and_grads

    def _static_clip(self, params_grads):
        params_and_grads = []
        sum_square_list = []
        with framework.name_scope('gradient_clip'):
            for p, g in params_grads:
                if g is None:
                    continue
                if self._need_clip_func is not None and not self._need_clip_func(
                        p):
                    continue
                merge_grad = g
                with p.block.program._optimized_guard([p, g]):
                    if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                        merge_grad = layers.merge_selected_rows(g)
                        merge_grad = layers.get_tensor_from_selected_rows(
                            merge_grad)

                    square = layers.square(merge_grad)
                    sum_square = layers.reduce_sum(input=square)
                    sum_square_list.append(sum_square)

            # all parameters have been filterd out
            if len(sum_square_list) == 0:
                return params_grads
            with p.block.program._optimized_guard([p, g]):
                global_norm_var = layers.sums(sum_square_list)
                global_norm_var = layers.sqrt(x=global_norm_var)
                max_global_norm = layers.fill_constant(
                    shape=[1], dtype="float32", value=self.clip_norm)
                scale_var = layers.elementwise_div(
                    x=max_global_norm,
                    y=layers.elementwise_max(
                        x=max_global_norm, y=global_norm_var))

            for p, g in params_grads:
                if g is None:
                    params_and_grads.append((p, g))
                    continue
                if self._need_clip_func is not None and not self._need_clip_func(
                        p):
                    params_and_grads.append((p, g))
                    continue

                with p.block.program._optimized_guard([p, g]):
                    new_grad = layers.elementwise_mul(x=g, y=scale_var)
                params_and_grads.append((p, new_grad))

        _correct_clip_op_role_var(params_and_grads)
        return params_and_grads


ClipByValue = GradClipByValue
ClipByNorm = GradClipByNorm
ClipByGlobalNorm = GradClipByGlobalNorm
