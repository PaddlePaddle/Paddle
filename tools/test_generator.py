# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import re
import functools
import warnings
import string

from six.moves import cStringIO
from paddle.fluid.proto import framework_pb2
from paddle.fluid.framework import OpProtoHolder, Variable
from paddle.fluid.layer_helper import LayerHelper


def _convert_(name):
    """
    Formatting.

    Args:
       name: The name/alias

    This function takes in a name and converts it to a standard format of
    group1_group2. Where as per the regular expression, group1 can have
    alphabets and numbers and group2 has capital alphabets.

    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _get_inputs(op_type):
    op_proto = OpProtoHolder.instance().get_op_proto(op_type)
    inputs = dict()
    for ipt in op_proto.inputs:
        inputs[ipt.name] = ""


def _get_outputs(op_type):
    op_proto = OpProtoHolder.instance().get_op_proto(op_type)
    outputs = {}
    for ipt in op_proto.outputs:
        outputs[ipt.name] = ""


def get_input_comments(op_type):
    return ""


def get_output_comments(op_type):
    return ""


def get_func_args(op_type):
    return ""


def get_inputs(op_type):
    return ""


def get_outputs(op_type):
    return ""


def get_op_py(op_type):
    input_comments = get_input_comments(op_type)
    output_comments = get_output_comments(op_type)
    args = get_func_args(op_type)
    inputs = get_inputs(op_type)
    outputs = get_outputs(op_type)

    code = """
\@templatedoc()
def {op_type}({args}):
    \"\"\"
    {op_type}
    
    Args:
        {input_comments}
    Returns:
        {output_comments}
    \"\"\"
    helper.append_op(
        type='{op_type}',
        {inputs},
        {outputs})    
""".format(
        input_comments=input_comments,
        output_comments=output_comments,
        args=args,
        op_type=op_type,
        inputs=inputs,
        outputs=outputs)

    return code


print(get_op_py("uniform_random_batch_size_like"))
#get_meta("linear_chain_crf")
