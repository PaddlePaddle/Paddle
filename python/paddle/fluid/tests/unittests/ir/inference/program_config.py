# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Optional, List, Callable, Dict, Any, Set
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle import compat as cpt
from paddle.fluid.initializer import NumpyArrayInitializer
from paddle.fluid.framework import convert_np_dtype_to_dtype_


class TensorConfig:
    '''
    A config builder for a input or a weight.
  
    InputVar's shape can be [-1, xxx], batch_size
    '''

    def __init__(self,
                 shape: [List[int]],
                 dtype: [str]="float32",
                 data: Optional[np.array]=None):
        '''
        shape: The shape of the tensor.
        dtype: The data type of the tensor.
        data: The value of WeightVar. for input, it should be None 
        '''
        self.shape = shape
        self.dtype = dtype
        self.data = data


class OpConfig:
    '''  A config builder for generating a Op.  '''

    def __init__(self,
                 type: str,
                 inputs: Dict[str, List[str]],
                 outputs: Dict[str, List[str]],
                 attrs: Dict[str, Any]):
        self.type = type
        self.inputs = inputs
        self.outputs = outputs
        self.attrs = attrs


class ProgramConfig:
    '''  A config builder for generating a Program.  '''

    def __init__(self,
                 ops: List[OpConfig],
                 weights: Dict[str, TensorConfig],
                 inputs: Dict[str, TensorConfig],
                 outputs: List[str]):
        self.ops = ops
        self.weights = weights
        self.inputs = inputs
        self.outputs = outputs


def create_fake_model(program_config):
    '''  Create a Paddle model(in memory) according to the given config.  '''
    paddle.enable_static()
    main_program_desc = core.ProgramDesc()
    util_program = fluid.Program()
    main_block_desc = main_program_desc.block(0)

    var_desc = main_block_desc.var(cpt.to_bytes("feed"))
    var_desc.set_type(core.VarDesc.VarType.FEED_MINIBATCH)
    var_desc.set_persistable(True)

    index = 0
    for name, tensor_config in program_config.inputs.items():
        var_desc = main_block_desc.var(cpt.to_bytes(name))
        var_desc.set_type(core.VarDesc.VarType.LOD_TENSOR)
        var_desc.set_dtype(convert_np_dtype_to_dtype_(tensor_config.dtype))
        var_desc.set_shape(tensor_config.shape)
        var_desc.set_need_check_feed(True)
        op_desc = main_block_desc._prepend_op()
        op_desc.set_type("feed")
        op_desc.set_input('X', ["feed"])
        op_desc.set_output('Out', [name])
        op_desc._set_attr("col", index)
        index = index + 1

    save_var_map = {}
    for name, tensor_config in program_config.weights.items():
        var_desc = main_block_desc.var(cpt.to_bytes(name))
        var_desc.set_type(core.VarDesc.VarType.LOD_TENSOR)
        var_desc.set_dtype(convert_np_dtype_to_dtype_(tensor_config.dtype))
        var_desc.set_shape(tensor_config.shape)
        var_desc.set_persistable(True)

        save_var_map[name] = util_program.global_block().create_parameter(
            dtype=tensor_config.dtype,
            shape=tensor_config.shape,
            type=core.VarDesc.VarType.LOD_TENSOR,
            name=name,
            initializer=NumpyArrayInitializer(tensor_config.data))
    in_vars = []
    for name in sorted(save_var_map.keys()):
        in_vars.append(save_var_map[name])

    out_var = util_program.global_block().create_var(
        type=core.VarDesc.VarType.RAW, name="out_var_0")
    out_var.desc.set_persistable(True)
    util_program.global_block().append_op(
        type='save_combine',
        inputs={'X': in_vars},
        outputs={'Y': out_var},
        attrs={'file_path': '',
               'save_to_memory': True})
    for op_config in program_config.ops:
        op_desc = main_block_desc.append_op()
        op_desc.set_type(op_config.type)
        for name, values in op_config.inputs.items():
            op_desc.set_input(name, values)
        for name, values in op_config.attrs.items():
            op_desc._set_attr(name, values)
        for name, values in op_config.outputs.items():
            op_desc.set_output(name, values)
            var_desc = main_block_desc.var(cpt.to_bytes(name))
            var_desc.set_type(core.VarDesc.VarType.LOD_TENSOR)
        op_desc.infer_var_type(main_block_desc)
        op_desc.infer_shape(main_block_desc)

    for index, name in enumerate(program_config.outputs):
        var_desc = main_block_desc.var(cpt.to_bytes("fetch"))
        var_desc.set_type(core.VarDesc.VarType.FETCH_LIST)
        var_desc.set_need_check_feed(True)
        op_desc = main_block_desc.append_op()
        op_desc.set_type("fetch")
        op_desc.set_input('X', [name])
        op_desc.set_output('Out', ["fetch"])
        op_desc._set_attr("col", index)

    main_program_desc._set_version()
    paddle.fluid.core.save_op_version_info(main_program_desc)

    model = main_program_desc.serialize_to_string()

    util_program._sync_with_cpp()
    place = fluid.CPUPlace()
    executor = fluid.Executor(place)
    scope = fluid.Scope()
    with fluid.scope_guard(scope):
        executor.run(util_program)
        params = scope.find_var("out_var_0").get_bytes()
    return model, params
