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

from paddle.incubate.cc.py_code_gen.ir.ir_op import Op


class PrimitiveOpExtractor:

    def Extract(self, ir_program):
        return [
            op
            for _, op in vars(ir_program).items()
            if isinstance(op, Op)
            if op.name not in self.GetInputOutputOpNames()
            if op.block_positional_arg_names is None
            if op.block_keyword_arg_names is None
            if len(op.GetResults()) > 0
        ]

    def GetInputOutputOpNames(self):
        return {
            "pd_op.data",
            "pd_op.feed",
            "builtin.parameter",
            "builtin.constant",
            "cf.yield",
            "builtin.shadow_output",
            "pd_op.fetch",
        }
