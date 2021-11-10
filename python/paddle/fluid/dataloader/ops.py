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
import paddle.fluid as fluid
import paddle.static as static

from paddle.fluid import core, framework
from paddle.fluid.layers.utils import _hash_with_id
from paddle.common_ops_import import *


__all__ = ["map"]


def map(map_func, inputs):
    assert not in_dygraph_mode(), \
            "paddle.io.map can only be used in static mode"
    helper = LayerHelper("map", **locals())

    # inputs are Variables hold LoDTensorBlockingQueue
    # TODO: cannot get tensor shape from LoDTensorBlockingQueue
    program_inputs = [static.data('input_{}'.format(i), [None]) for i in range(len(inputs))]

    # build map program
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with static.guard(main_program, startup_program):
        program_outputs = map_func(*program_inputs)
    
    input_var_names = [v.name for v in program_inputs]
    output_var_names = [v.name for v in program_outputs]

    global_block = self._main_program.desc.block(0)
    program_id = _hash_with_id(main_program, map_func)

    outputs = \
        [helper.create_variable(
            name=unique_name.generate("map"),
            type=core.VarDesc.VarType.LOD_TENSOR_BLOCKING_QUEUE,
            persistable=True) for _ in range(len(program_outputs))]
    attrs = {
        "global_block": global_block,
        "program_id": program_id,
        "start_op_index": 0,
        "end_op_index": global_block.op_size(),
        "input_var_names": input_var_names,
        "output_var_names": output_var_names
    }

    helper.append_op(
        type="map",
        inputs={"X": inputs},
        outputs={"Out": outputs},
        attrs=attrs)

    return outputs
