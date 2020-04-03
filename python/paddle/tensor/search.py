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
from __future__ import print_function
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype

# TODO: define searching & indexing functions of a tensor
# __all__ = ['argmax',
#            'argmin',
#            'argsort',
#            'has_inf',
#            'has_nan',
#            'masked_select',
#            'topk',
#            'where',
#            'index_select',
#            'nonzero',
#            'sort']

__all__ = ['index_sample']


def index_sample(x, index):
    """
    **IndexSample Layer**

    IndexSample OP returns the element of the specified location of X, 
    and the location is specified by Index. 

    .. code-block:: text


                Given:

                X = [[1, 2, 3, 4, 5],
                     [6, 7, 8, 9, 10]]

                Index = [[0, 1, 3],
                         [0, 2, 4]]

                Then:

                Out = [[1, 2, 4],
                       [6, 8, 10]]

    Args:
        x (Variable): The source input tensor with 2-D shape. Supported data type is 
            int32, int64, float32, float64.
        index (Variable): The index input tensor with 2-D shape, first dimension should be same with X. 
            Data type is int32 or int64.

    Returns:
        output (Variable): The output is a tensor with the same shape as index.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            # create x value
            x_shape = (2, 5)
            x_type = "float64"
            x_np = np.random.random(x_shape).astype(x_type)

            # create index value
            index_shape = (2, 3)
            index_type = "int32"
            index_np = np.random.randint(low=0, 
                                         high=x_shape[1],
                                         size=index_shape).astype(index_type)

            x = fluid.data(name='x', shape=[-1, 5], dtype='float64')
            index = fluid.data(name='index', shape=[-1, 3], dtype='int32')
            output = paddle.index_sample(x=x, index=index)

    """
    helper = LayerHelper("index_sample", **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'paddle.tensor.search.index_sample')
    check_variable_and_dtype(index, 'index', ['int32', 'int64'],
                             'paddle.tensor.search.index_sample')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='index_sample',
        inputs={'X': x,
                'Index': index},
        outputs={'Out': out})
    return out
