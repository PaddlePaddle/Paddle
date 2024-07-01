import tensorrt as trt
import numpy as np
import paddle
import paddle.nn.functional as F

# Only Work in TRT 10.x

class PaddlePhiPlugin(trt.IPluginV2DynamicExt):
    def __init__(self, op_name):
        super(PaddlePhiPlugin, self).__init__()
        self.op_name = op_name

    def get_plugin_type(self):
        return "PaddlePhiPlugin"

    def get_plugin_version(self):
        return "1"

    def get_field_names(self):
        return []

    def initialize(self):
        return 0

    def terminate(self):
        pass

    def get_serialization_size(self):
        return len(self.op_name.encode('utf-8'))

    def serialize(self):
        return self.op_name.encode('utf-8')

    def destroy(self):
        pass

    def clone(self):
        return PaddlePhiPlugin(self.op_name)

    def get_output_dimensions(self, index, inputs, nb_input_dims, context):
        return inputs[0]

    def supports_format_combination(self, pos, in_out, num_inputs, num_outputs):
        return in_out[pos].format == trt.TensorFormat.LINEAR and in_out[pos].dtype == trt.float32

    def configure_plugin(self, in_tensors, out_tensors, in_desc, out_desc):
        pass

    def enqueue(self, batch_size, inputs, outputs, workspace, stream):
        input_tensor = paddle.to_tensor(np.array(inputs[0]))
        output_tensor = paddle.to_tensor(np.array(outputs[0]))

        if self.op_name == "relu":
            result = F.relu(input_tensor)
        elif self.op_name == "sigmoid":
            result = paddle.sigmoid(input_tensor)
        else:
            raise NotImplementedError(f"Operation {self.op_name} is not implemented.")

        output_tensor.copy_(result)
        return 0

    def get_plugin_namespace(self):
        return ""

    def set_plugin_namespace(self, namespace):
        pass

class PaddlePhiPluginCreator(trt.IPluginCreator):
    def __init__(self):
        super(PaddlePhiPluginCreator, self).__init__()

    def get_plugin_name(self):
        return "PaddlePhiPlugin"

    def get_plugin_version(self):
        return "1"

    def get_field_names(self):
        return []

    def create_plugin(self, name, field_data):
        return PaddlePhiPlugin(name)

    def deserialize_plugin(self, name, serial_data):
        return PaddlePhiPlugin(serial_data.decode('utf-8'))

    def get_plugin_namespace(self):
        return ""

    def set_plugin_namespace(self, namespace):
        pass

trt.init_libnvinfer_plugins(None, "")
plugin_creator = PaddlePhiPluginCreator()
trt.get_plugin_registry().register_creator(plugin_creator, "")

def build_engine(op_name):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    
    input_tensor = network.add_input(name="input", dtype=trt.float32, shape=(-1,))
    plugin_layer = network.add_plugin_v2([input_tensor], PaddlePhiPlugin(op_name))
    network.mark_output(plugin_layer.get_output(0))

    return builder.build_engine(network, config)

if __name__ == "__main__":
    op_name = "relu"
    engine = build_engine(op_name)