# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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

import logging
import numpy as np
import paddle
from paddle import base
from paddle import pir
from paddle.base.log_helper import get_logger
from paddle.base.core import get_value_shape_range_info
from util import run_pir_pass, map_dtype
from custom_plugin import PaddlePhiPluginCreator, GENERAL_PLUGIN_OPS_LIST
from register import converter_registry
from impls.core import  *



_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)

class PaddleToTensorRTConverter:
    def __init__(self, paddle_program, scope):
        try:
            import tensorrt as trt
        except Exception:
            _logger.info("import tensorrt failed, you may install it via `python3 -m pip install --upgrade tensorrt` according to https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html")
        self.scope = scope
        self.program = paddle_program
        params = paddle_program.global_block().all_parameters()
        param_dict = {}
        # save parameters
        for v in params:
            name = v.get_defining_op().attrs()["parameter_name"]
            weight_array = np.array(self.scope.var(name).get_tensor())
            # weights = trt.Weights(weight_array)
            param_dict.update({name: weight_array})
        self.param_dict = param_dict
        paddle.framework.set_flags({"FLAGS_enable_collect_shape": True})
       
    def find_graph_inputs_outputs(self, group_op):
        operations = list(group_op.blocks())[0].ops
        all_values = {}
        output_values = {}
        
        graph_output_values = []
    
        def __is_output_value(value):
            for op in value.all_used_ops():
                if op.name() == "cf.yield":
                    return True
            return False

        # Collect all output values from all operations
        for op in operations:
            for result in op.results():
                output_values[result.id] = result
                all_values[result.id] = result
                if __is_output_value(result):
                    graph_output_values.append(result)
            for operand in op.operands():
                source = operand.source()
                all_values[source.id] = source

        # Input values are those that are in all_values but not in output_values
        input_values = [value for value_id, value in all_values.items() if value_id not in output_values]

        return input_values, graph_output_values

    def convert_subgraph_to_trt(self, program, group_op):
        operations = list(group_op.blocks())[0].ops
        input_values, output_values = self.find_graph_inputs_outputs(group_op)
        builder = trt.Builder(trt.Logger(trt.Logger.VERBOSE))
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        
        weights = {}
        
        # Mapping from Value id to TensorRT ITensor
        value_to_trt_tensor = {}
        min_shape_map = {}
        opt_shape_map = {}
        max_shape_map = {}
        input_names = []
        for value in input_values:
            defining_op = value.get_defining_op()
            if defining_op.name() == "builtin.parameter":
                param_name = defining_op.attrs()["parameter_name"]
                weight = trt.Weights(self.param_dict[param_name])
                weights[value.id] = weight
                value_to_trt_tensor[value.id] = weight
                input_names.append("")
            else:
                shape = value.shape
                dtype = map_dtype(value.dtype.name)
                min_shape = get_value_shape_range_info(
                    value, False, paddle.base.core.ShapeMode.kMIN
                )
                opt_shape = get_value_shape_range_info(
                    value, False, paddle.base.core.ShapeMode.kOPT
                )
                max_shape = get_value_shape_range_info(
                    value, False, paddle.base.core.ShapeMode.kMAX
                )
                
                input_name = f"input_{value.id}"
                input_tensor = network.add_input(name=input_name, dtype=dtype, shape=shape)
                _logger.info(f"set min_shape of {value} as {min_shape}")
                _logger.info(f"set opt_shape of {value} as {opt_shape}")
                _logger.info(f"set max_shape of {value} as {max_shape}")
                profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
                min_shape_map[input_name] = min_shape
                opt_shape_map[input_name] = opt_shape
                max_shape_map[input_name] = max_shape
                input_names.append(input_name)
                value_to_trt_tensor[value.id] = input_tensor
           
        for op in operations:
            operands = [value_to_trt_tensor[operand.source().id] for operand in op.operands()]
            layer = self.convert(network, op, operands)

            for idx, result in enumerate(op.results()):
                value_to_trt_tensor[result.id] = layer.get_output(idx)
        out_shapes = []
        out_names = []
        out_types = []
        for result_value in output_values:
            # identity_out = network.add_identity(value_to_trt_tensor[result_value.id]).get_output(0)
            output_tensor = value_to_trt_tensor[result_value.id]
            network.mark_output(output_tensor)
            out_names.append(output_tensor.name)
            out_shapes.append(result_value.shape)
            out_types.append(result_value.dtype)
        
        config = builder.create_builder_config()
        config.add_optimization_profile(profile)
        trt_engine = builder.build_engine(network, config)
        trt_params = paddle.base.libpaddle.TRTEngineParams()
        trt_params.min_input_shape = min_shape_map
        trt_params.max_input_shape = max_shape_map
        trt_params.optim_input_shape = opt_shape_map
        CACHE_FILE = "./engine.trt"
        with open(CACHE_FILE, "wb") as f:
            f.write(trt_engine.serialize())
        trt_params.engine_serialized_data = CACHE_FILE
        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            program
        ):
            pir.set_insertion_point(group_op)
            out = paddle._C_ops.tensorrt_engine(
                input_values,
                trt_params,
                input_names,
                out_names,
                out_shapes,
                out_types
            )

        return out

    def convert(self, network, paddle_op, inputs):
        op_name = paddle_op.name()
        if op_name in ["cf.yield"]:
            return
        elif op_name in GENERAL_PLUGIN_OPS_LIST:
            out = network.add_plugin_v2(inputs)
        else:
            converter_func = converter_registry.get(op_name)
            if converter_func is None:
                raise NotImplementedError(f"Converter for {op_name} not implemented.")
            out = converter_func(network, paddle_op, inputs)
        return out
    
    def convert_program_to_trt(self):
        for op in self.program.global_block().ops:
            if op.name() == "cinn_op.group":
                print(f"!!! start process {op.name()}")
                new_out = self.convert_subgraph_to_trt(self.program, op)
                orin_out_values = op.results()
                for o_i in range(len(orin_out_values)):
                    orin_out_values[o_i].replace_all_uses_with(new_out[o_i])
                self.program.global_block().remove_op(op)