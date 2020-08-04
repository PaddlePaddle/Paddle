# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.framework import Operator
from paddle.fluid import core


def create_op_struct(op):
    if not isinstance(op, Operator):
        raise TypeError("Not support use {} init OpStruct".format(op.type))
    return OpStruct(op.type, op.inputs, op.outputs, op.attrs)

def get_origin_op_in_block(op_struct, block):
    for op in block.ops:
        if op.type == op_struct.type and \
            op.inputs == op_struct.inputs and \
            op.outputs == op_struct.outputs and \
            op.attrs == op_struct.attrs:
            return op
    raise ValueError("Not find OP {} in Given Block".format(op_struct))

class OpStruct(object):
    """
    Record op properties of Operateor in python
    """
    def __init__(self, 
                 type=None,
                 inputs=None,
                 outputs=None,
                 attrs=None):
        self.type=type
        self.inputs=inputs,
        self.outputs=outputs,
        self.attrs=attrs

    def __str__(self):
        op_str = "OP --> Out: [ {} ] = Type[ {} ]( Input: [ {} ] ) with Attrs: [ {} ] ".format(self.outputs, self.type, self.inputs, self.attrs)
        return op_str

    def __eq__(self, other):
        """
        Overrides the default implementation
        """
        if isinstance(other, OpStruct):
            return self.type == other.type and \
                    str(self.inputs) == str(other.inputs) and \
                    str(self.outputs) == str(other.outputs) and \
                    str(self.attrs) == str(other.attrs)
        else:
            raise TypeError("OpStruct can't compare with {}, its type {}".format(other, other.type))

    def __lt__(self, other):
        if isinstance(other, OpStruct):
            return self.type < other.type
        else:
            raise TypeError("OpStruct can't compare with {}, its type {}".format(other, other.type))