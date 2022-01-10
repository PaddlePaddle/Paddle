# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

from functools import reduce

import paddle
from paddle.fluid.framework import dygraph_only, _dygraph_tracer, _varbase_creator
from paddle import _C_ops


#input==output, inplace strategy of reshape has no cost almostly
def _inplace_reshape_dygraph(x, shape):
    x_shape = _varbase_creator(dtype=x.dtype)
    _dygraph_tracer().trace_op(
        type="reshape2",
        inputs={'X': x},
        outputs={'Out': x,
                 'XShape': x_shape},
        attrs={'shape': shape},
        stop_gradient=True)


@dygraph_only
def parameters_to_vector(parameters, name=None):
    """
    Flatten parameters to a 1-D Tensor.

    Args:
        parameters(Iterable[Tensor]): Iterable Tensors that are trainable parameters of a Layer.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A 1-D Tensor, which represents the parameters of a Layer.
    

    Examples:
       .. code-block:: python

            import paddle
            linear = paddle.nn.Linear(10, 15)

            paddle.nn.utils.parameters_to_vector(linear.parameters())
            # 1-D Tensor: [165]

    """
    dtype = parameters[0].dtype
    origin_shapes = []
    for param in parameters:
        origin_shapes.append(param.shape)
        _inplace_reshape_dygraph(param, [-1])

    out = _varbase_creator(dtype=dtype)
    _dygraph_tracer().trace_op(
        type='concat',
        inputs={'X': parameters},
        outputs={'Out': [out]},
        attrs={'axis': 0},
        stop_gradient=True)
    for i, param in enumerate(parameters):
        _inplace_reshape_dygraph(param, origin_shapes[i])
    return out


@dygraph_only
def vector_to_parameters(vec, parameters, name=None):
    """
    Transform a 1-D Tensor to the input ``parameters`` .

    Args:
        vec (Tensor): A 1-D Tensor, which will be sliced and copied to the input ``parameters`` .
        parameters (Iterable[Tensor]): Iterable Tensors that are trainable parameters of a Layer.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Examples:
       .. code-block:: python

            import paddle
            weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(3.))
            linear1 = paddle.nn.Linear(10, 15, weight_attr)

            vec = paddle.nn.utils.parameters_to_vector(linear1.parameters())

            linear2 = paddle.nn.Linear(10, 15)
            # copy weight of linear1 to linear2
            paddle.nn.utils.vector_to_parameters(vec, linear2.parameters())
            # weight: Tensor(shape=[10, 15], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
            #                 [[3. , ..., 3. ],
            #                  [..., ..., ...],
            #                  [3. , ..., 3. ]])
    """
    origin_shapes = []
    sections = []
    for param in parameters:
        shape = param.shape
        origin_shapes.append(shape)
        numel = reduce(lambda x, y: x * y, shape)
        sections.append(numel)

    _dygraph_tracer().trace_op(
        type='split',
        inputs={'X': [vec]},
        outputs={'Out': parameters},
        attrs={'axis': 0,
               'sections': sections},
        stop_gradient=True)

    for i, param in enumerate(parameters):
        _inplace_reshape_dygraph(param, origin_shapes[i])
    return
