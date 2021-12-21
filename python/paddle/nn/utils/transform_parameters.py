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
from paddle.fluid.framework import Variable, in_dygraph_mode
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle import _C_ops


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
    vec_list = []
    if in_dygraph_mode():
        for param in parameters:
            vec, _ = _C_ops.reshape2(param, None, 'shape', [-1])
            vec_list.append(vec)
        return _C_ops.concat(vec_list, 'axis', 0)

    helper = LayerHelper("parameters_to_vector", **locals())
    param_dtype = parameters[0].dtype
    for id, param in enumerate(parameters):
        check_variable_and_dtype(
            param, 'parameters[{}]'.format(id),
            ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
            "parameters_to_vector")
        if param.dtype != param_dtype:
            raise TypeError(
                "All the Tensors in the parameters must have the same data type."
            )
        vec = helper.create_variable_for_type_inference(dtype=param_dtype)
        x_shape = helper.create_variable_for_type_inference(dtype=param_dtype)
        # use View strategy that don't have Tensor Copy
        helper.append_op(
            type='reshape2',
            inputs={'X': param},
            outputs={'Out': vec,
                     'XShape': x_shape},
            attrs={'shape': [-1]})
        vec_list.append(vec)

    param_vec = helper.create_variable_for_type_inference(dtype=param_dtype)
    helper.append_op(
        type='concat',
        inputs={'X': vec_list},
        outputs={'Out': param_vec},
        attrs={'axis': 0})
    return param_vec


def vector_to_parameters(vec, parameters, name=None):
    """
    Transform a Tensor with 1-D shape to the parameters.

    Args:
        vec (Tensor): A Tensor with 1-D shape, which represents the parameters of a Layer.
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
    start = 0
    helper = LayerHelper("vector_to_parameters", **locals())
    if in_dygraph_mode():
        with paddle.no_grad():
            for param in parameters:
                shape = param.shape
                numel = reduce(lambda x, y: x * y, shape)
                end = start + numel
                slice_data = _C_ops.slice(vec, None, None, 'axes', [0],
                                          'infer_flags', [1], 'starts',
                                          [start], 'ends', [end])
                _C_ops.reshape2_(slice_data, None, 'shape', shape)
                helper.append_op(
                    type='assign',
                    inputs={'X': slice_data},
                    outputs={'Out': param})
                start += numel
            return

    check_variable_and_dtype(
        vec, 'x', ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
        "vector_to_parameters")
    assert len(vec.shape) == 1, "'vec' must be a Tensor with 1-D shape."

    for param in parameters:
        shape = param.shape
        numel = reduce(lambda x, y: x * y, shape)
        end = start + numel

        slice_data = helper.create_variable_for_type_inference(
            dtype=param.dtype)
        helper.append_op(
            type='slice',
            inputs={'Input': vec},
            outputs={'Out': slice_data},
            attrs={
                'axes': [0],
                'infer_flags': [1],
                'starts': [start],
                'ends': [end]
            })

        # avoid backward for parameters
        slice_data.stop_gradient = True
        x_shape = helper.create_variable_for_type_inference(dtype=param.dtype)
        out = helper.create_variable_for_type_inference(dtype=param.dtype)

        # use Inplace strategy that don't have Tensor Copy
        helper.append_op(
            type='reshape2',
            inputs={'X': slice_data},
            outputs={'Out': slice_data,
                     'XShape': x_shape},
            attrs={'shape': shape})

        helper.append_op(
            type='assign', inputs={'X': slice_data}, outputs={'Out': param})
        start += numel
