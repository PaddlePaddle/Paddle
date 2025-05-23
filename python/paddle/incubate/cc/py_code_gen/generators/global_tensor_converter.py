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

import os

from paddle.incubate.cc.py_code_gen.ir.ir_tensor import Tensor


class GlobalTensorConverter:

    def __init__(self):
        self.local_name_prefix2seq_no = {}
        self.global_name2local_tensor = {}
        self.enable_local_tensor = os.getenv(
            "ATHENA_ENABLE_LOCAL_TENSOR"
        ) not in {
            "0",
            "False",
            "false",
            "OFF",
        }

    def ConvertToLocalTensor(self, tensor, prefix=None):
        if not self.enable_local_tensor:
            return tensor
        if tensor is None:
            return None
        if tensor.name not in self.global_name2local_tensor:
            prefix = tensor.local_name_prefix if prefix is None else prefix
            self.global_name2local_tensor[tensor.name] = Tensor(
                local_name_prefix=prefix,
                name=f"{prefix}_{self._GetLocalNameSeqNo(prefix)}",
                arg_name_as_input=tensor.arg_name_as_input,
                defining_op_name=tensor.defining_op_name,
                type=tensor.type,
                dim_exprs=tensor.dim_exprs,
            )
        return self.global_name2local_tensor[tensor.name]

    def _GetLocalNameSeqNo(self, local_name_prefix):
        if local_name_prefix not in self.local_name_prefix2seq_no:
            self.local_name_prefix2seq_no[local_name_prefix] = 0
        local_name_seq_no = self.local_name_prefix2seq_no[local_name_prefix]
        self.local_name_prefix2seq_no[local_name_prefix] += 1
        return local_name_seq_no
