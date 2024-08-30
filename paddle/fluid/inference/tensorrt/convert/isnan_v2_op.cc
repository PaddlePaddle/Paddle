/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * IsnanV2 Op
 */
class IsnanV2OpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a isnan_v2 op to tensorrt layer";
    framework::OpDesc op_desc(op, nullptr);
    std::string input_x_name = op_desc.Input("X").front();
    std::string output_name = op_desc.Output("Out").front();
    auto* input_x_tensor = engine_->GetITensor(input_x_name);
#if IS_TRT_VERSION_GE(10100)
    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, Unary, *input_x_tensor, nvinfer1::UnaryOperation::kISNAN);
    ReplenishLayerAndOutput(layer, "isnan_v2", {output_name}, test_mode);
#else
    auto* equal_layer =
        TRT_ENGINE_ADD_LAYER(engine_,
                             ElementWise,
                             *input_x_tensor,
                             *input_x_tensor,
                             nvinfer1::ElementWiseOperation::kEQUAL);
    auto* layer = TRT_ENGINE_ADD_LAYER(engine_,
                                       Unary,
                                       *equal_layer->getOutput(0),
                                       nvinfer1::UnaryOperation::kNOT);
    ReplenishLayerAndOutput(layer, "isnan_v2", {output_name}, test_mode);
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
REGISTER_TRT_OP_CONVERTER(isnan_v2, IsnanV2OpConverter);
