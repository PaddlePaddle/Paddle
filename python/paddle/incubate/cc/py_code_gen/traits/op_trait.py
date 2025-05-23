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

from paddle.incubate.cc.py_code_gen.ir import ir_op


class OpTrait:

    def Op(
        self,
        name,
        op_id,
        input_types,
        output_types,
        attrs,
        block_positional_arg_names=None,
        block_keyword_arg_names=None,
        block_positional_arg_types=None,
        block_keyword_arg_types=None,
    ):
        if name == "pd_kernel.phi_kernel":
            name = attrs["op_name"].value
        return ir_op.Op(
            name=name,
            op_id=op_id,
            input_types=input_types,
            output_types=output_types,
            attrs=attrs,
            block_positional_arg_names=block_positional_arg_names,
            block_keyword_arg_names=block_keyword_arg_names,
            block_positional_arg_types=block_positional_arg_types,
            block_keyword_arg_types=block_keyword_arg_types,
            __operands_symbols_signature__=attrs[
                "__operands_symbols_signature__"
            ],
            __results_symbols_signature__=attrs[
                "__results_symbols_signature__"
            ],
        )
