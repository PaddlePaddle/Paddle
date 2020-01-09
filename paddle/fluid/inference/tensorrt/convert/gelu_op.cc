/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/plugin/gelu_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * Gelu converter from fluid to tensorRT.
 */
class GeluOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert fluid gelu op to tensorrt gelu layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    int input_num = op_desc.Input("X").size();
    PADDLE_ENFORCE_EQ(input_num, 1,
                      platform::errors::InvalidArgument(
                          "gelu op has only 1 input, but got %d", input_num));
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    // Get output
    size_t output_num = op_desc.Output("Out").size();
    PADDLE_ENFORCE_EQ(output_num, 1,
                      platform::errors::InvalidArgument(
                          "gelu op has only 1 output, but got %d", output_num));
    // Get input shape and volume
    nvinfer1::Dims input_shape = input->getDimensions();
    size_t input_volume = 1;
    for (int i = 0; i < input_shape.nbDims; i++) {
      input_volume *= input_shape.d[i];
    }
    plugin::GeluPlugin* plugin = new plugin::GeluPlugin(input_volume);
    nvinfer1::IPluginLayer* layer =
        engine_->AddPlugin(&input, input_num, plugin);
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "gelu", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(gelu, GeluOpConverter);
