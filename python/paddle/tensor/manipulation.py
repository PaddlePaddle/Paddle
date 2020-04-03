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

from ..fluid.layers import core
from ..fluid.layer_helper import LayerHelper
from ..fluid.framework import Variable, OpProtoHolder, in_dygraph_mode, convert_np_dtype_to_dtype_
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype

# TODO: define functions to manipulate a tensor  
__all__ = [
    #            'cast',
    #            'concat',
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
    'flip',
    #            'unbind',
    #            'roll'
]


def flip(input, dims, name=None):
    """

    Reverse the order of a n-D tensor along given axis in dims.

    Args:
        input (Variable): A Tensor(or LoDTensor) with shape :math:`[N_1, N_2,..., N_k]` . The data type of the input Tensor
            should be float32, float64, int32, int64, bool.
        dims (list): The axis to flip on.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: Tensor or LoDTensor calculated by flip layer. The data type is same with input.

    Examples:
        .. code-block:: python

          import paddle
          import paddle.fluid as fluid
          import numpy as np
          input = fluid.data(name="x", shape=[-1, 2, 2], dtype='float32')
          output = paddle.flip(input, dims=[0, 1])
          exe = fluid.Executor(fluid.CPUPlace())
          exe.run(fluid.default_startup_program())
          img = np.arange(12).reshape((3,2,2)).astype(np.float32)
          res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
          print(res) # [[[10,11][8, 9]],[[6, 7],[4, 5]] [[2, 3],[0, 1]]]
    """
    if in_dygraph_mode():
        inputs = {'X': [input]}
        outs = core.ops.flip(inputs, {'dims': dims})
        return outs['Out'][0]

    helper = LayerHelper("flip", **locals())
    check_type(input, 'X', (Variable), 'flip')
    dtype = helper.input_dtype()
    check_dtype(dtype, 'X',
                ['float16', 'float32', 'float64', 'int32', 'int64', 'bool'],
                'flip')
    check_type(dims, 'dims', (list, tuple), 'flip')
    assert len(dims) > 0, 'len(dims) must be greater than 0.'
    if name is None:
        out = helper.create_variable_for_type_inference(dtype)
    else:
        out = helper.create_variable(name=name, dtype=dtype, persistable=False)

    helper.append_op(
        type="flip",
        inputs={"X": input},
        outputs={"Out": out},
        attrs={"dims": dims})
    return out
