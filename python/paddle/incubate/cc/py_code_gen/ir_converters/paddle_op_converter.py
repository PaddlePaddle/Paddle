# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.incubate.cc.py_code_gen.ir import paddle_op
from paddle.incubate.cc.py_code_gen.ir_converters.paddle_attr_converter import (
    ConvertAttributeToString,
)


def ConvertToPaddleOp(op):
    return paddle_op.Op(
        name=op.name,
        op_id=op.op_id,
        input_types=op.input_types,
        output_types=op.output_types,
        attrs={
            name: ConvertAttributeToString(attr)
            for name, attr in op.attrs.items()
        },
        block_positional_arg_names=op.block_positional_arg_names,
        block_keyword_arg_names=op.block_keyword_arg_names,
        block_positional_arg_types=op.block_positional_arg_types,
        block_keyword_arg_types=op.block_keyword_arg_types,
        base_op=op,
        __operands_symbols_signature__=op.__operands_symbols_signature__,
        __results_symbols_signature__=op.__results_symbols_signature__,
    )
