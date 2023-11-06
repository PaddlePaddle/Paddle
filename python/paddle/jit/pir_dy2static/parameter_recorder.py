# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.base import framework

from ..dy2static.program_translator import _program_hash, synchronized


class ParametersRecorder:
    def __init__(self):
        self.params_dict = {}
        self.tensor2opresult = {}

    @synchronized
    def get(self, program, tensor):
        from paddle.pir.core import create_parameter, vartype_to_datatype

        """use the default_program as key, append tensor the parameter list."""
        key = _program_hash(program)
        if key not in self.params_dict:
            self.params_dict[key] = set()
            self.tensor2opresult[key] = {}

        params = self.params_dict[key]
        mappings = self.tensor2opresult[key]
        if id(tensor) not in mappings:
            non_used_initializer = paddle.nn.initializer.Constant(0.0)
            op_result = create_parameter(
                dtype=vartype_to_datatype[tensor.dtype],
                shape=tensor.shape,
                type=tensor.type,
                initializer=non_used_initializer,
            )
            if isinstance(tensor, framework.EagerParamBase):
                params.add(tensor)
            mappings[id(tensor)] = op_result
        return mappings[id(tensor)]

    def pop(self, program):
        hash_id = _program_hash(program)
        params = self.params_dict.get(hash_id)
        if params is None:
            return [], []
        params_values = [
            self.tensor2opresult[hash_id][id(x)] for x in list(params)
        ]
        del self.params_dict[hash_id]
        del self.tensor2opresult[hash_id]
        return list(params), list(params_values)


_global_parameter_recorder = ParametersRecorder()
