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

import numpy as np
import paddle
import tensorrt as trt
from paddle import base
from paddle import pir
from paddle.base.core import get_value_shape_range_info
from util import run_pir_pass, map_dtype
from custom_plugin import PaddlePhiPluginCreator, GENERAL_PLUGIN_OPS_LIST
from register import converter_registry
from impls.core import  *
paddle.framework.set_flags({"FLAGS_enable_collect_shape": True})
  
class PaddleToTensorRTConverter:
    def __init__(self, paddle_program, scope):
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
        
        # input_tensor = self.network.add_input(name="input", dtype=trt.float32, shape=input_shape)
       
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

    def convert_subgraph_to_trt(self, group_op):
        operations = list(group_op.blocks())[0].ops
        input_values, output_values = self.find_graph_inputs_outputs(group_op)
        builder = trt.Builder(trt.Logger(trt.Logger.VERBOSE))
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        
        weights = {}
        
        # Mapping from Value id to TensorRT ITensor
        value_to_trt_tensor = {}
        for value in input_values:
            import pdb;pdb.set_trace()
            defining_op = value.get_defining_op()
            if defining_op.name() == "builtin.parameter":
                param_name = defining_op.attrs()["parameter_name"]
                weight = trt.Weights(self.param_dict[param_name])
                weights[value.id] = weight
                value_to_trt_tensor[value.id] = weight
            # elif defining_op.name() == "pd_op.data":
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
                # min_shape = tuple(dim if dim != -1 else 1 for dim in shape)  # Minimum shape for dynamic dimensions
                # opt_shape = tuple(dim if dim != -1 else 4 for dim in shape)  # Optimal shape for dynamic dimensions
                # max_shape = tuple(dim if dim != -1 else 8 for dim in shape)  # Maximum shape for dynamic dimensions
                input_name = f"input_{value.id}"
                input_tensor = network.add_input(name=input_name, dtype=dtype, shape=shape)
                profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
                value_to_trt_tensor[value.id] = input_tensor

                
        for op in operations:
            operands = [value_to_trt_tensor[operand.source().id] for operand in op.operands()]
            layer = self.convert(network, op, operands)

            for idx, result in enumerate(op.results()):
                value_to_trt_tensor[result.id] = layer.get_output(idx)
        out_shapes = []
        for result_value in output_values:
            network.mark_output(value_to_trt_tensor[result_value.id])
            out_shapes.append(result_value.shape)
        
        config = builder.create_builder_config()
        config.add_optimization_profile(profile)
        trt_engine = builder.build_engine(network, config)
        trt_params = paddle.base.libpaddle.TRTEngineParams()
        trt_params.min_input_shape = {"x": [1, 1]}
        trt_params.max_input_shape = {"x": [10, 1]}
        trt_params.optim_input_shape = {"x": [5, 1]}
        trt_params.engine_serialized_data = str(trt_engine)
        out = paddle._C_ops.tensorrt_engine(
            input_values,
            trt_params,
            ["x"],
            ["out"],
            out_shapes,
            # [[1, 1]],
            [paddle.base.libpaddle.DataType.FLOAT32],
        )
        paddle.
    

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
                trt_engine = self.convert_subgraph_to_trt(op)
                print(trt_engine)
                # create fake tensorrt op
                

def main():
    from util import get_dummy_program
    program, scope, param_dict = get_dummy_program()

    with paddle.pir_utils.IrGuard():
        with paddle.static.program_guard(
            program
        ):
            exe = paddle.static.Executor()
            input_array = np.random.randn(1, 64).astype('float32')
            out = exe.run(
                program,
                feed={"input": input_array},
                fetch_list=program.list_vars()[-1]
            )
            print("first time, the out shape is: ", out[0].shape)

            input_array2 = np.random.randn(8, 64).astype('float32')
            out = exe.run(
                program,
                feed={"input": input_array2},
                fetch_list=program.list_vars()[-1]
            )
            print("second time, the out shape is: ", out[0].shape)

    program = run_pir_pass(program, partition_mode=True)
    converter = PaddleToTensorRTConverter(program, scope)
    converter.convert_program_to_trt()

if __name__ == "__main__":
    main()