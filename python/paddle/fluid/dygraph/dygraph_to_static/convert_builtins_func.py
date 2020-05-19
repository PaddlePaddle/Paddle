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

from paddle.fluid import framework
from paddle.fluid import core
from paddle.fluid import unique_name


def converted_len(var):
    # return variable(length) from shape ops based on var.type
    if isinstance(var, framework.Variable):
        block = current_block(var)
        out = create_new_var(block, 'int32', name='shape')
        if var.type in [
                core.VarDesc.VarType.LOD_TENSOR,
                core.VarDesc.VarType.SELECTED_ROWS
        ]:
            # Note: Length of var may be known ahead of time in dygraph,
            # but it probably represents batch size which can be variant.
            # so we return a variable dynamically inferred from var.shape.
            block.append_op(
                type='shape', inputs={'Input': var}, outputs={'Out': out})
        elif var.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY:
            block.append_op(
                type='lod_array_length',
                inputs={'X': [var]},
                outputs={'Out': [out]})
            return out
        else:
            raise TypeError(
                'len(var) only supports LoDTensor/LoDTensorArray/SelectedRows, but received %s.'
                % type(var))
        # return shape[0] as length.
        return out[0]
    else:
        return builtin_len(var)


def builtin_len(x):
    return len(x)


def current_block(var):
    return var.block.program.current_block()


def create_new_var(block, dtype, name):
    tmp_name = unique_name.generate(name)
    return block.create_var(name=tmp_name, dtype=dtype)
