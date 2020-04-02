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

# TODO: define functions to manipulate a tensor  
from __future__ import print_function

import numpy as np
import warnings
import six
import os
import inspect
from ..fluid.layer_helper import LayerHelper
from ..fluid.initializer import Normal, Constant, NumpyArrayInitializer
from ..fluid.framework import Variable, OpProtoHolder, in_dygraph_mode, dygraph_only, _dygraph_tracer, default_main_program
from ..fluid import dygraph_utils
from ..fluid.param_attr import ParamAttr
from ..fluid import unique_name
from ..fluid import core
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype

__all__ = [
    #            'cast',
    'concat',
    #            'expand',
    #            'expand_as',
    #            'flatten',
    #            'gather',
    #            'gather_nd',
    #            'reshape',
    #            'reverse',
    #            'scatter',
    #            'scatter_nd_add',
    #            'scatter_nd',
    #            'shard_index',
    #            'slice',
    #            'split',
    #            'squeeze',
    #            'stack',
    #            'strided_slice',
    #            'transpose',
    #            'unique',
    #            'unique_with_counts',
    #            'unsqueeze',
    #            'unstack',
    #            'flip',
    #            'unbind',
    #            'roll'
]


def concat(input, axis=0, name=None):
    """
    **Concat**
    This OP concatenates the input along the axis.
    Args:
        input(list): List of input Tensors with data type float32, float64, int32,
            int64.
        axis(int32|Variable, optional):  A scalar with type ``int32`` or a ``Tensor`` with shape [1] and type ``int32``. Axis to compute indices along. The effective range
            is [-R, R), where R is Rank(x). when axis<0, it works the same way
            as axis+R. Default is 0.
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    Returns:
        Variable: A Tensor with the same data type as input's.
    Examples:
        .. code-block:: python
            import paddle.tensor as tensor
            import paddle.fluid as fluid
            import numpy as np
            in1 = np.array([[1,2,3],
                            [4,5,6]])
            in2 = np.array([[11,12,13],
                            [14,15,16]])
            in3 = np.array([[21,22],
                            [23,24]])
            with fluid.dygraph.guard():
                x1 = fluid.dygraph.to_variable(in1)
                x2 = fluid.dygraph.to_variable(in2)
                x3 = fluid.dygraph.to_variable(in3)
                out1 = tensor.concat(input=[x1,x2,x3], axis=-1)
                out2 = tensor.concat(input=[x1,x2], axis=0)
                print(out1.numpy())
                # [[ 1  2  3 11 12 13 21 22]
                #  [ 4  5  6 14 15 16 23 24]]
                print(out2.numpy())
                # [[ 1  2  3]
                #  [ 4  5  6]
                #  [11 12 13]
                #  [14 15 16]]
    """

    if in_dygraph_mode():
        if isinstance(axis, Variable):
            axis = axis.numpy()
            assert axis.shape == (
                1, ), "axis of type Variable should have shape [1]"
            axis = axis[0]
        return core.ops.concat(input, 'axis', axis)

    if not isinstance(input, list):
        warnings.warn(
            "The type of input in concat should be list, but received %s." %
            (type(input)))
        input = [input]
    for id, x in enumerate(input):
        check_variable_and_dtype(
            x, 'input[' + str(id) + ']',
            ['float16', 'float32', 'float64', 'int32', 'int64'], 'concat')
    check_type(axis, 'axis', (int, Variable), 'concat')

    helper = LayerHelper('concat', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())

    if input[0].desc.type() == core.VarDesc.VarType.LOD_TENSOR_ARRAY:
        assert len(input) == 1, "If the elements of 'input' in concat are Variable(LoDTensorArray), " \
                            "number of the elements must be 1, but received %s." % len(x)
        out_index = helper.create_variable_for_type_inference(dtype="int32")
        helper.append_op(
            type='tensor_array_to_tensor',
            inputs={'X': input[0]},
            outputs={'Out': [out],
                     'OutIndex': [out_index]},
            attrs={'axis': axis,
                   'use_stack': False})
    else:
        inputs = {'X': input}
        attrs = {}
        if isinstance(axis, Variable):
            axis.stop_gradient = True
            inputs['AxisTensor'] = axis
        else:
            attrs['axis'] = axis

        helper.append_op(
            type='concat', inputs=inputs, outputs={'Out': [out]}, attrs=attrs)
    return out
