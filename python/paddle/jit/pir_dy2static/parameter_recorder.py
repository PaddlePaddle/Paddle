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
from paddle.autograd.backward_utils import ValueDict
from paddle.framework import core

from ..dy2static.program_translator import _program_hash, synchronized


class ParametersRecorder:
    def __init__(self):
        self.params_dict = {}
        self.tensor2value = {}

    @synchronized
    def get(self, program, tensor):
        from paddle.pir.core import create_parameter, vartype_to_datatype

        """use the default_program as key, append tensor the parameter list."""
        key = _program_hash(program)
        if key not in self.params_dict:
            self.params_dict[key] = set()
            self.tensor2value[key] = {}

        params = self.params_dict[key]
        mappings = self.tensor2value[key]
        if id(tensor) not in mappings:
            non_used_initializer = paddle.nn.initializer.Constant(0.0)
            dtype = tensor.dtype
            if isinstance(dtype, core.VarDesc.VarType):
                vartype_to_datatype[dtype]
            value = create_parameter(
                dtype=dtype,
                shape=tensor.shape,
                type=tensor.type,
                name=tensor.name,
                initializer=non_used_initializer,
                trainable=(not tensor.stop_gradient),
                placements=tensor.placements,
                process_mesh=tensor.process_mesh,
            )

            if isinstance(tensor, paddle.Tensor):
                params.add(tensor)
            mappings[id(tensor)] = value

        return mappings[id(tensor)]

    def pop(self, program):
        hash_id = _program_hash(program)
        params = self.params_dict.get(hash_id)
        if params is None:
            return [], []
        params = list(params)
        params.sort(key=lambda x: x.name)
        params_values = [self.tensor2value[hash_id][id(x)] for x in params]
        del self.params_dict[hash_id]
        del self.tensor2value[hash_id]
        return params, params_values


class InplaceMap:
    def __init__(self):
        self.params_dict = {}

    @synchronized
    def add(self, program, origin_value, new_value):
        key = _program_hash(program)
        if key not in self.params_dict:
            self.params_dict[key] = ValueDict()
        inplace_dict = self.params_dict[key]
        inplace_dict[origin_value] = new_value

    def get(self, program, value):
        inplace_dict = self.params_dict.get(_program_hash(program))
        if inplace_dict is None:
            return None
        if value not in inplace_dict:
            return None
        root_var = inplace_dict[value]
        saved = []
        while root_var in inplace_dict:
            saved.append(root_var)
            root_var = inplace_dict[root_var]
        for var in saved:
            inplace_dict[var] = root_var
        return root_var

    def restore_checkpoint(self, checkpoint):
        # InplaceMap is a nested effect.
        # when enter a block, we should save a checkpoint
        # when exit a block, we should restore a checkpoint
        # for example:
        # if cond > 0:
        #    x [:] = 0
        # return x
        # x[:] only effect current cond block, we should restore in false block.
        self.params_dict = checkpoint

    def save_checkpoint(self):
        ckeckpoint = {}
        for program_id, params in self.params_dict.items():
            new_params = dict(params.items())
            ckeckpoint[program_id] = new_params
        return ckeckpoint


_global_parameter_recorder = ParametersRecorder()
_global_inplace_map = InplaceMap()
