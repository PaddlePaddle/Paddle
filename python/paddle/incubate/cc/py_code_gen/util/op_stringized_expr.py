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

from dataclasses import dataclass

from paddle.incubate.cc.py_code_gen.util.hash_combine import hash_combine


@dataclass
class OpStringizedExpr:
    op_name: str
    op_expr: str
    input_name_prefix: str
    num_results: int

    def __hash__(self):
        hash_value = hash(self.op_name)
        hash_value = hash_combine(hash_value, hash(self.op_expr))
        hash_value = hash_combine(hash_value, hash(self.input_name_prefix))
        hash_value = hash_combine(hash_value, hash(self.num_results))
        return hash_value
