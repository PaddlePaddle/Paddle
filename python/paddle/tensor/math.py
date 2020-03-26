#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
math functions
"""

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers.nn import _elementwise_op, _elementwise_op_in_dygraph
from paddle.fluid.framework import in_dygraph_mode

# TODO: define math functions  
__all__ = [
    'abs', 'acos', 'asin', 'atan', 'ceil', 'cos', 'cumsum', 'elementwise_add',
    'elementwise_div', 'elementwise_floordiv', 'elementwise_max',
    'elementwise_min', 'elementwise_mod', 'elementwise_mul', 'elementwise_pow',
    'elementwise_sub', 'exp', 'floor', 'increment', 'log', 'mul', 'multiplex',
    'pow', 'reciprocal', 'reduce_max', 'reduce_min', 'reduce_prod',
    'reduce_sum', 'round', 'rsqrt', 'scale', 'sign', 'sin', 'sqrt', 'square',
    'stanh', 'sum', 'sums', 'tanh', 'elementwise_sum', 'max', 'min', 'mm',
    'div', 'add', 'atan', 'logsumexp', 'inverse', 'log1p', 'erf', 'addcmul',
    'addmm'
]


def div(x, y, out=None, name=None):
    """
    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            def gen_data():
                return {
                    "x": np.array([2, 3, 4]).astype('float32'),
                    "y": np.array([1, 5, 2]).astype('float32')
                }

            x = fluid.data(name="x", shape=[3], dtype='float32')
            y = fluid.data(name="y", shape=[3], dtype='float32')
            z = fluid.layers.elementwise_div(x, y)
            # z = x / y

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            z_value = exe.run(feed=gen_data(),
                                fetch_list=[z.name])

            print(z_value) # [2., 0.6, 2.]


        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            def gen_data():
                return {
                    "x": np.ones((2, 3, 4, 5)).astype('float32'),
                    "y": np.zeros((3, 4)).astype('float32')
                }

            x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
            y = fluid.data(name="y", shape=[3,4], dtype='float32')
            z = fluid.layers.elementwise_div(x, y, axis=1)
            # z = x / y

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            z_value = exe.run(feed=gen_data(),
                                fetch_list=[z.name])

            print(z_value) # z.shape=[2,3,4,5]


        ..  code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            def gen_data():
                return {
                    "x": np.random.randint(1, 5, size=[2, 3, 4, 5]).astype('float32'),
                    "y": np.random.randint(1, 5, size=[5]).astype('float32')
                }

            x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
            y = fluid.data(name="y", shape=[5], dtype='float32')
            z = fluid.layers.elementwise_div(x, y, axis=3)
            # z = x / y

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            z_value = exe.run(feed=gen_data(),
                                fetch_list=[z.name])
            print(z_value) # z.shape=[2,3,4,5]

    """

    op_type = 'elementwise_div'
    axis = -1
    act = None
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)

    return _elementwise_op(LayerHelper(op_type, **locals()))
