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
#include "paddle/fluid/inference/tensorrt/plugin/swish_op_plugin.h"

namespace nvinfer1 {
class ILayer;
}  // namespace nvinfer1
namespace paddle {
namespace framework {
class Scope;

namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

class SwishOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert fluid swish op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    int input_num = op_desc.Input("X").size();
    PADDLE_ENFORCE_EQ(input_num,
                      1,
                      platform::errors::InvalidArgument(
                          "The input X's size must equal to 1 in TRT swish op."
                          " But received X's size %d.",
                          input_num));
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    // Get output
    size_t output_num = op_desc.Output("Out").size();
    PADDLE_ENFORCE_EQ(
        output_num,
        1UL,
        platform::errors::InvalidArgument(
            "The output Out's size must equal to 1 in TRT swish op. "
            "But received Out's size %u.",
            output_num));
    // Get attrs
    float beta = PADDLE_GET_CONST(float, op_desc.GetAttr("beta"));

    nvinfer1::ILayer* layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      int32_t rank = input->getDimensions().nbDims;
      nvinfer1::Dims constant_shape;
      constant_shape.nbDims = rank;
      std::fill(constant_shape.d, constant_shape.d + rank, 1);
      std::vector<float> weight_data{beta};
      auto* beta_data = AddConstantLayer(weight_data.data(), constant_shape);
      auto* input_mul_with_beta = Prod(beta_data, input);
      auto* sigmoid = TRT_ENGINE_ADD_LAYER(engine_,
                                           Activation,
                                           *input_mul_with_beta,
                                           nvinfer1::ActivationType::kSIGMOID);
      layer = TRT_ENGINE_ADD_LAYER(engine_,
                                   ElementWise,
                                   *input,
                                   *(sigmoid->getOutput(0)),
                                   nvinfer1::ElementWiseOperation::kPROD);
    } else {
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      plugin::SwishPlugin* plugin = new plugin::SwishPlugin(beta, with_fp16);
      layer = engine_->AddPluginV2Ext(&input, input_num, plugin);
    }

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "swish", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(swish, SwishOpConverter);
