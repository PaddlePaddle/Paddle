#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import os
from .layer_function_generator import generate_layer_fn, generate_activation_fn
from .. import core
from ..framework import convert_np_dtype_to_dtype_

__activations_noattr__ = [
    'sigmoid',
    'logsigmoid',
    'exp',
    'tanh',
    'atan',
    'tanh_shrink',
    'sqrt',
    'rsqrt',
    'abs',
    'ceil',
    'floor',
    'cos',
    'acos',
    'asin',
    'sin',
    'round',
    'reciprocal',
    'square',
    'softplus',
    'softsign',
]

__all__ = []

for _OP in set(__all__):
    globals()[_OP] = generate_layer_fn(_OP)

# It is a hot fix in some unittest using:
#   fluid.layers.scale(x=x, scale=10.0, out=out_var)
# e.g.: test_program_code.py, test_dist_train.py
globals()['_scale'] = generate_layer_fn('scale')

globals()['_elementwise_div'] = generate_layer_fn('elementwise_div')

__all__ += __activations_noattr__

for _OP in set(__activations_noattr__):
    globals()[_OP] = generate_activation_fn(_OP)

__all__ += ['softshrink']

_softshrink_ = generate_layer_fn('softshrink')


def softshrink(x, alpha=None):
    locals_var = locals().copy()
    kwargs = dict()
    for name, val in locals_var.items():
        if val is not None:
            if name == 'alpha':
                kwargs['lambda'] = val
            else:
                kwargs[name] = val
    return _softshrink_(**kwargs)


softshrink.__doc__ = """
:strong:`Softshrink Activation Operator`

..  math::
    out = \begin{cases}
            x - \alpha, \text{if } x > \alpha \\
            x + \alpha, \text{if } x < -\alpha \\
            0,  \text{otherwise}
            \end{cases}


Args:
    x: Input of Softshrink operator
    alpha (FLOAT): non-negative offset
    
Returns:
    Output of Softshrink operator

Examples:
    .. code-block:: python
    
        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[784])
        result = fluid.layers.softshrink(x=data, alpha=0.3)
"""

__all__ += ['hard_shrink']

_hard_shrink_ = generate_layer_fn('hard_shrink')


def hard_shrink(x, threshold=None):
    locals_var = locals().copy()
    kwargs = dict()
    for name, val in locals_var.items():
        if val is not None:
            kwargs[name] = val
    return _hard_shrink_(**kwargs)


hard_shrink.__doc__ = _hard_shrink_.__doc__ + """
Examples:

    >>> import paddle.fluid as fluid
    >>> data = fluid.layers.data(name="input", shape=[784])
    >>> result = fluid.layers.hard_shrink(x=data, threshold=0.3)
"""

__all__ += ['cumsum']

_cum_sum_ = generate_layer_fn('cumsum')


def cumsum(x, axis=None, exclusive=None, reverse=None):
    locals_var = locals().copy()
    kwargs = dict()
    for name, val in locals_var.items():
        if val is not None:
            kwargs[name] = val
    return _cum_sum_(**kwargs)


cumsum.__doc__ = _cum_sum_.__doc__ + """
Examples:

    >>> import paddle.fluid as fluid
    >>> data = fluid.layers.data(name="input", shape=[32, 784])
    >>> result = fluid.layers.cumsum(data, axis=0)
"""

__all__ += ['thresholded_relu']

_thresholded_relu_ = generate_layer_fn('thresholded_relu')


def thresholded_relu(x, threshold=None):
    locals_var = locals().copy()
    kwargs = dict()
    for name, val in locals_var.items():
        if val is not None:
            kwargs[name] = val

    return _thresholded_relu_(**kwargs)


thresholded_relu.__doc__ = """
:strong:`Thresholded ReLU Activation Operator`

Equation:
    ..  math::
        out = \\begin{cases}
            x, &if x > threshold \\\\
            0, &otherwise
            \\end{cases}

Args:
    x(Variable): The input of Thresholded ReLU op, Tensor or LoDTensor, dtype: float32 or float64.
        
    threshold(float, optional): The threshold value. Note that if the arg `threshold` is not set, the threshold in the equation is 1.0.

Returns:

    Variable: The output of Thresholded ReLU op, Tensor or LoDTensor, dtype: float32 or float64, the same as the input, shape: the same as the input.

Examples:
    
    .. code-block:: python
    
        # declarative mode
        import numpy as np
        from paddle import fluid
        
        x = fluid.data(name="x", shape=(-1, 3), dtype="float32")
        y = fluid.layers.thresholded_relu(x, threshold=0.1)
        
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        start = fluid.default_startup_program()
        main = fluid.default_main_program()
        
        data = np.random.randn(2, 3).astype("float32")
        exe.run(start)
        
        y_np, = exe.run(main, feed={"x": data}, fetch_list=[y])
        
        data
        # array([[ 0.21134382, -1.1805999 ,  0.32876605],
        #        [-1.2210793 , -0.7365624 ,  1.0013918 ]], dtype=float32)
        y_np
        # array([[ 0.21134382, -0.        ,  0.32876605],
        #        [-0.        , -0.        ,  1.0013918 ]], dtype=float32)

    .. code-block:: python
    
        # imperative mode
        import numpy as np
        from paddle import fluid
        import paddle.fluid.dygraph as dg
        
        data = np.random.randn(2, 3).astype("float32")
        place = fluid.CPUPlace()
        with dg.guard(place) as g:
            x = dg.to_variable(data)
            y = fluid.layers.thresholded_relu(x, threshold=0.1)
            y_np = y.numpy()
        data
        # array([[ 0.21134382, -1.1805999 ,  0.32876605],
        #        [-1.2210793 , -0.7365624 ,  1.0013918 ]], dtype=float32)
        y_np
        # array([[ 0.21134382, -0.        ,  0.32876605],
        #        [-0.        , -0.        ,  1.0013918 ]], dtype=float32)
"""
