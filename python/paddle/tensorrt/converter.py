import tensorrt as trt
import numpy as np
import paddle
from paddle import base
from paddle import pir
from .util import run_pir_pass, map_dtype
  
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
            defining_op = value.get_defining_op()
            if defining_op.name() == "builtin.parameter":
                param_name = defining_op.attrs()["parameter_name"]
                weight = trt.Weights(self.param_dict[param_name])
                weights[value.id] = weight
                value_to_trt_tensor[value.id] = weight
            elif defining_op.name() == "pd_op.data":
                shape = value.shape
                dtype = map_dtype(value.dtype.name)
                min_shape = tuple(dim if dim != -1 else 1 for dim in shape)  # Minimum shape for dynamic dimensions
                opt_shape = tuple(dim if dim != -1 else 4 for dim in shape)  # Optimal shape for dynamic dimensions
                max_shape = tuple(dim if dim != -1 else 8 for dim in shape)  # Maximum shape for dynamic dimensions
                input_name = f"input_{value.id}"
                input_tensor = network.add_input(name=input_name, dtype=dtype, shape=shape)
                profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
                value_to_trt_tensor[value.id] = input_tensor
                
        for op in operations:
            operands = [value_to_trt_tensor[operand.source().id] for operand in op.operands()]
            layer = self.convert(network, op, operands)

            for idx, result in enumerate(op.results()):
                value_to_trt_tensor[result.id] = layer.get_output(idx)
        for result_value in output_values:
            network.mark_output(value_to_trt_tensor[result_value.id])
        
        config = builder.create_builder_config()
        config.add_optimization_profile(profile)
        return builder.build_engine(network, config)
    

    def convert(self, network, paddle_op, inputs):
        if paddle_op.name() == "pd_op.add" or  paddle_op.name() == "pd_op.elementwise_add":
            weight_shape = paddle_op.operands()[1].source().shape
            weight_tensor = network.add_constant(weight_shape, inputs[1]).get_output(0)
            
            weight_shape = paddle_op.operands()[0].source().shape
            new_bias_shape = weight_shape
            reshape_layer = network.add_reshape(weight_tensor, new_bias_shape).get_output(0) 

            out = network.add_elementwise(inputs[0], reshape_layer, trt.ElementWiseOperation.SUM)
            return out
        elif paddle_op.name() == "pd_op.relu":
            out = network.add_activation(inputs[0], trt.ActivationType.RELU)
            return out
        elif paddle_op.name() == "pd_op.matmul":
            weight_shape = paddle_op.operands()[1].source().shape
            weight_tensor = network.add_constant(weight_shape, inputs[1]).get_output(0)
            out = network.add_matrix_multiply(inputs[0], trt.MatrixOperation.NONE, weight_tensor, trt.MatrixOperation.NONE)
            return out
        elif paddle_op.name() == "pd_op.conv2d":
            strides = paddle_op.attrs()["strides"]
            padding = paddle_op.attrs()["paddings"]
            dilations = paddle_op.attrs()["dilations"]
            actvation = paddle_op.attrs()["actvation"]
            groups = paddle_op.attrs()["groups"]
            out = network.add_convolution(
                inputs[0],
                weight_shape,
                weight_tensor,
                stride=strides,
                padding=padding,
                groups=groups,
                dilations=dilations,
                actvation=actvation
            )
            return out
        elif paddle_op.name() == "cf.yield":
            pass
        elif paddle_op.name() in GENERAL_PLUGIN_OPS_LIST:
            out = network.add_plugin
        else:
            raise NotImplementedError(f"Conversion for {paddle_op} not implemented.")
    
    def convert_program_to_trt(self):
        for op in self.program.global_block().ops:
            if op.name() == "cinn_op.group":
                trt_engine = self.convert_subgraph_to_trt(op)
                print(trt_engine)


def run_pir_pass(program):
    pm = pir.PassManager(opt_level=4)
    pm.enable_print_statistics()
    pm.enable_ir_printing()
    for pass_item in pass_attr_list:
        for pass_name, pass_attr in pass_item.items():
            pm.add_pass(pass_name, pass_attr)
    pm.run(program)
    return program


        
def main():
    program, scope, param_dict = get_program()
    converter = PaddleToTensorRTConverter(program, scope)
    converter.convert_program_to_trt()

if __name__ == "__main__":
    main()