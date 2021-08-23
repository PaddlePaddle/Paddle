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

from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.fluid.contrib.slim.quantization import QuantizationFreezePass
from paddle.fluid.framework import IrGraph, IrNode, Operator
from paddle.fluid.executor import global_scope


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


def create_quant_model(model,
                       params,
                       activation_quantize_type='moving_average_abs_max',
                       weight_quantize_type='channel_wise_abs_max',
                       save=False):
    place = paddle.CUDAPlace(0)
    scope = global_scope()
    exe = paddle.static.Executor(place)
    [inference_program, feed_target_names,
     fetch_targets] = paddle.static.load_inference_model(
         path_prefix=None,
         executor=exe,
         model_filename=model,
         params_filename=params)
    graph = IrGraph(core.Graph(inference_program.desc), for_test=True)

    transform_pass = QuantizationTransformPass(
        scope=scope,
        place=place,
        activation_quantize_type=activation_quantize_type,
        weight_quantize_type=weight_quantize_type)
    transform_pass.apply(graph)

    out_scale_op_list = [
        "conv2d",
        "depthwise_conv2d",
        "mul",
        "matmul",
        "relu",
        "leaky_relu",
        "relu6",
        "sigmoid",
        "tanh",
        "prelu",
        "swish",
        "softmax",
        "batch_norm",
        "layer_norm",
        "elementwise_add",
        "pool2d",
        "reshape2",
        "transpose2",
        "concat",
        "elementwise_mul",
        "scale",
        "slice",
        "hard_swish",
        "hard_sigmoid",
        "conv2d_transpose",
        "gru",
        "bilinear_interp",
        "nearest_interp",
        "trilinear_interp",
        "flatten",
        "flatten2",
        "transpose",
        "pad2d",
        "reshape",
        "layer_norm",
    ]
    op_real_in_out_name = {
        "conv2d": [["Input", "Filter"], ["Output"]],
        "depthwise_conv2d": [["Input", "Filter"], ["Output"]],
        "conv2d_transpose": [["Input", "Filter"], ["Output"]],
        "mul": [["X", "Y"], ["Out"]],
        "matmul": [["X", "Y"], ["Out"]],
        "pool2d": [["X"], ["Out"]],
        "elementwise_add": [["X", "Y"], ["Out"]],
        "concat": [["X"], ["Out"]],
        "softmax": [["X"], ["Out"]],
        "argmax": [["X"], ["Out"]],
        "transpose": [["X"], ["Out"]],
        "equal": [["X", "Y"], ["Out"]],
        "gather": [["X"], ["Out"]],
        "greater_equal": [["X", "Y"], ["Out"]],
        "greater_than": [["X", "Y"], ["Out"]],
        "less_equal": [["X", "Y"], ["Out"]],
        "less_than": [["X", "Y"], ["Out"]],
        "mean": [["X"], ["Out"]],
        "not_equal": [["X", "Y"], ["Out"]],
        "reshape": [["X"], ["Out"]],
        "reshape2": [["X"], ["Out"]],
        "transpose2": [["X"], ["Out"]],
        "bilinear_interp": [["X"], ["Out"]],
        "nearest_interp": [["X"], ["Out"]],
        "trilinear_interp": [["X"], ["Out"]],
        "slice": [["Input"], ["Out"]],
        "squeeze": [["X"], ["Out"]],
        "elementwise_sub": [["X", "Y"], ["Out"]],
        "relu": [["X"], ["Out"]],
        "relu6": [["X"], ["Out"]],
        "leaky_relu": [["X"], ["Out"]],
        "prelu": [["X"], ["Out"]],
        "tanh": [["X"], ["Out"]],
        "swish": [["X"], ["Out"]],
        "dropout": [["X"], ["Out"]],
        "batch_norm": [["X"], ["Y"]],
        "layer_norm": [["X"], ["Y"]],
        "sigmoid": [["X"], ["Out"]],
        "elementwise_mul": [["X", "Y"], ["Out"]],
        "scale": [["X"], ["Out"]],
        "hard_swish": [["X"], ["Out"]],
        "hard_sigmoid": [["X"], ["Out"]],
        "gru": [["Input", "Weight"], ["Hidden"]],
        "lstm": [["Input", "Weight"], ["Hidden"]],
        "pad2d": [["X"], ["Out"]],
        "flatten": [["X"], ["Out"]],
        "flatten2": [["X"], ["Out"]],
    }

    def _get_op_output_var_names(op):
        """ """
        assert isinstance(op, (IrNode, Operator)), \
            "The input op should be IrNode or Operator."
        var_names = []
        op_name = op.name() if isinstance(op, IrNode) \
            else op.type
        if op_name not in op_real_in_out_name:
            return []

        name_list = op_real_in_out_name[op_name][1]
        for name in name_list:
            var_name = op.output(name)
            if isinstance(var_name, list):
                var_names.extend(var_name)
            else:
                var_names.append(var_name)
        return var_names

    op_nodes = graph.all_op_nodes()
    for op_node in op_nodes:
        if op_node.name() in out_scale_op_list:
            var_names = _get_op_output_var_names(op_node)
            for var_name in var_names:
                in_node = graph._find_node_by_name(op_node.outputs, var_name)
                if in_node.dtype() not in \
                    [core.VarDesc.VarType.FP64, core.VarDesc.VarType.FP32]:
                    continue

                op_node.op()._set_attr("out_threshold", 3.0)

    # Freeze graph for inference, but the weight of fc/conv is still float type.
    freeze_pass = QuantizationFreezePass(
        scope=scope, place=place, weight_quantize_type=weight_quantize_type)
    freeze_pass.apply(graph)

    main_program = graph.to_program()

    # modify fake_quantize_moving_average_abs_max(InScale) and fake_channel_wise_dequantize_max_abs(Scales)
    op_nodes = graph.all_op_nodes()
    for op_node in op_nodes:
        if op_node.name() == 'fake_quantize_moving_average_abs_max':
            var_name = op_node.input("InScale")[0]
            tensor = scope.var(var_name).get_tensor()
            tensor.set(np.array([1], dtype=np.float32), place)
        elif op_node.name() == 'fake_channel_wise_dequantize_max_abs':
            var_name = op_node.input("Scales")[0]
            tensor = scope.var(var_name).get_tensor()
            tensor.set(np.ones(tensor.shape(), dtype=np.float32), place)

    if save:
        fluid.io.save_inference_model(
            'test_inference_model',
            feed_target_names,
            fetch_targets,
            exe,
            main_program=main_program)

    feed_vars = [
        main_program.global_block().var(name) for name in feed_target_names
    ]
    serialized_program = paddle.static.serialize_program(
        feed_vars, fetch_targets, program=main_program)
    serialized_params = paddle.static.serialize_persistables(
        feed_vars, fetch_targets, executor=exe, program=main_program)
    return serialized_program, serialized_params
