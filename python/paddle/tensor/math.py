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

# TODO: define math functions  
__all__ = [#'abs',
    #            'acos',
    #            'asin',
    #            'atan',
    #            'ceil',
    #            'cos',
    #            'cumsum',
    #            'elementwise_add',
    #            'elementwise_div',
    #            'elementwise_floordiv',
    #            'elementwise_max',
    #            'elementwise_min',
    #            'elementwise_mod',
    #            'elementwise_mul',
    #            'elementwise_pow',
    #            'elementwise_sub',
    #            'exp',
    #            'floor',
    #            'increment',
    #            'log',
    #            'mul',
    #            'multiplex',
    #            'pow',
    #            'reciprocal',
    #            'reduce_max',
    #            'reduce_min',
    #            'reduce_prod',
    #            'reduce_sum',
    #            'round',
    #            'rsqrt',
    #            'scale',
    #            'sign',
    #            'sin',
    #            'sqrt',
    #            'square',
    #            'stanh',
    #            'sum',
    #            'sums',
    #            'tanh',
    #            'elementwise_sum',
    #            'max',
    #            'min',
    #            'mm',
           'div',
    #            'add',
    #            'atan',
    #            'logsumexp',
    #            'inverse',
    #            'log1p',
    #            'erf',
    #            'addcmul',
    #            'addmm'
            ]

from paddle.common_ops_import import *


def _elementwise_op(helper):
    op_type = helper.layer_type
    x = helper.kwargs.get('x', None)
    y = helper.kwargs.get('y', None)

    assert x is not None, 'x cannot be None in {}'.format(op_type)
    assert y is not None, 'y cannot be None in {}'.format(op_type)
    check_variable_and_dtype(
        x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'], op_type)
    check_variable_and_dtype(
        y, 'y', ['float16', 'float32', 'float64', 'int32', 'int64'], op_type)

    axis = helper.kwargs.get('axis', -1)
    use_mkldnn = helper.kwargs.get('use_mkldnn', False)
    name = helper.kwargs.get('name', None)
    out = helper.kwargs.get('out', None)
    if out is None:
        if name is None:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
        else:
            out = helper.create_variable(
                name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type=op_type,
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs={'axis': axis,
               'use_mkldnn': use_mkldnn})
    return helper.append_activation(out)


def div(x, y, out=None, name=None):
    """
    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.tensor as tensor
            import numpy as np

            def gen_data():
                return {
                    "x": np.array([2, 3, 4]).astype('float32'),
                    "y": np.array([1, 5, 2]).astype('float32')
                }

            x = fluid.data(name="x", shape=[3], dtype='float32')
            y = fluid.data(name="y", shape=[3], dtype='float32')
            z = tensor.div(x, y)
            # z = x / y

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            z_value = exe.run(feed=gen_data(),
                                fetch_list=[z.name])

            print(z_value) # [2., 0.6, 2.]


        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.tensor as tensor
            import numpy as np

            def gen_data():
                return {
                    "x": np.ones((2, 3, 4, 5)).astype('float32'),
                    "y": np.zeros((4, 5)).astype('float32')
                }

            x = fluid.data(name="x", shape=[2, 3, 4, 5], dtype='float32')
            y = fluid.data(name="y", shape=[4, 5], dtype='float32')
            z = tensor.div(x, y, name='z')
            # z = x / y

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            z_value = exe.run(feed=gen_data(),
                                fetch_list=[z.name])

            print(z_value[0])
            print(z_value[0].shape) # z.shape=[2,3,4,5]


        ..  code-block:: python

            import paddle.fluid as fluid
            import paddle.tensor as tensor
            import numpy as np

            def gen_data():
                return {
                    "x": np.random.randint(1, 5, size=[2, 3, 4, 5]).astype('float32'),
                    "y": np.random.randint(1, 5, size=[5]).astype('float32')
                }

            x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
            y = fluid.data(name="y", shape=[5], dtype='float32')
            z = tensor.div(x, y)
            # z = x / y

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            z_value = exe.run(feed=gen_data(),
                                fetch_list=[z.name])
            print(z_value[0])
            print(z_value[0].shape) # z.shape=[2,3,4,5]
    """
    op_type = 'elementwise_div'
    axis = -1
    act = None
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)
    if name and out:
        warnings.warn(
            "Both name and out parameters have been set in fluid.layers.%s(), only out will take effect to specify the result storage. "
            "You can discard either one to solve this warning." % op_type,
            category=UserWarning,
            stacklevel=2)
    return _elementwise_op(LayerHelper(op_type, **locals()))
