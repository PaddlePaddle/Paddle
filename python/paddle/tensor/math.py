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

import warnings

from ..fluid.framework import OpProtoHolder, core, in_dygraph_mode
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype
from ..fluid.layers.layer_function_generator import _generate_doc_string_

# TODO: define math functions
# yapf: disable
__all__ = [
#            'abs',
#            'acos',
#            'asin',
           'atan',
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
           'sin',
           'sqrt',
#            'square',
#            'stanh',
#            'sum',
#            'sums',
           'tanh',
#            'elementwise_sum',
#            'max',
#            'min',
#            'mm',
#            'div',
#            'add',
#            'atan',
#            'logsumexp',
#            'inverse',
#            'log1p',
#            'erf',
#            'addcmul',
#            'addmm']
]
# yapf: enable.


def generate_op_noattr(op_type):
    """Register the Python layer for an Operator without Attribute..

    Args:
       op_type: The name of the operator to be created.

    This function takes in the operator type (sin, tanh etc) and
    creates the operator functionality.

    """
    op_proto = OpProtoHolder.instance().get_op_proto(op_type)

    def func(x, name=None, out=None):
        if in_dygraph_mode():
            inputs = {'X': [x]}
            op = getattr(core.ops, op_type)
            outs = op(inputs)
            return outs['Out'][0]

        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 op_type)
        helper = LayerHelper(op_type, **locals())

        if name and out:
            warnings.warn(
                "Both name and out parameters have been set in fluid.tensor.math.%s(), only out will take effect to specify the result storage. "
                "You can discard either one to solve this warning." % op_type,
                category=UserWarning,
                stacklevel=2)
        if not out:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type=op_type, inputs={"X": x}, outputs={"Out": out})
        return out

    func.__name__ = op_type
    func.__doc__ = _generate_doc_string_(
        op_proto,
        additional_args_lines=[
            "name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`.\n    "
            "out(Variable, optional): The default value is None. Optional output can be any created Variable that meets the requirements to store the result of operation. if out is None, a new Varibale will be create to store the result."
        ])
    func.__doc__ = func.__doc__ + """

Return type
  Variable
Examples:
    .. code-block:: python

        import numpy as np
        
        import paddle
        import paddle.fluid as fluid

        inputs = fluid.data(name="x", shape = [None, 4], dtype='float32')
        output = paddle.tensor.math.%s(inputs)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        #input.shape=1X4, batch_size=1
        img = np.array([[1.0, 2.0, 3.0, 4.0]]).astype(np.float32)
        res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
        print(res)
""" % op_type
    return func


__ops__noattr__ = [
    'atan',
    'sin',
    'sqrt',
    'tanh',
]

for _OP in set(__ops__noattr__):
    globals()[_OP] = generate_op_noattr(_OP)
