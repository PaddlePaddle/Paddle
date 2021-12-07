#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from ...fluid import core, framework, Program, program_guard, unique_name
from ...fluid.layers.utils import _hash_with_id
from ...common_ops_import import *


__all__ = ["map"]


def _to_list(l):
    if isinstance(l, (list, tuple)):
        return l
    return [l]


class MapGuard(object):
    def __init__(self, main_program):
        if not isinstance(main_program, Program):
            raise TypeError("MapGuard should init with a Program")
        self._main_program = main_program

    def __enter__(self):
        self._main_program._create_block()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._main_program._rollback()
        return exc_type is None


def map(map_func, inputs):
    assert not in_dygraph_mode(), \
            "paddle.io.map can only be used in static mode"
    helper = LayerHelper("map", **locals())

    # build map block
    main_program = helper.main_program
    with MapGuard(main_program):
        program_id = _hash_with_id(main_program, map_func)
        map_block = main_program.current_block()

        inputs = _to_list(inputs)
        program_inputs = [
            map_block.create_var(
                name=unique_name.generate("map_sub"),
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False) for i in range(len(inputs))]
        program_outputs = map_func(*program_inputs)
        program_outputs = _to_list(program_outputs)
    
        input_var_names = [v.name for v in program_inputs]
        output_var_names = [v.name for v in program_outputs]

    outputs = \
        [helper.create_variable(
            name=unique_name.generate("map"),
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=True) for _ in range(len(program_outputs))]
    attrs = {
        "map_block": map_block,
        "program_id": program_id,
        "input_var_names": input_var_names,
        "output_var_names": output_var_names
    }

    helper.append_op(
        type="map",
        inputs={"In": inputs},
        outputs={"Out": outputs},
        attrs=attrs)

    return outputs
