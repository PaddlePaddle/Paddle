/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

class TanhshrinkOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert fluid Tanhshrink op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    int input_num = op_desc.Input("X").size();
    PADDLE_ENFORCE_EQ(
        input_num,
        1,
        platform::errors::InvalidArgument(
            "The input X's size must equal to 1 in TRT Tanhshrink op."
            " But received X's size %d.",
            input_num));
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    // Get output
    size_t output_num = op_desc.Output("Out").size();
    PADDLE_ENFORCE_EQ(
        output_num,
        1UL,
        platform::errors::InvalidArgument(
            "The output Out's size must equal to 1 in TRT Tanhshrink op. "
            "But received Out's size %u.",
            output_num));

    nvinfer1::ILayer* layer = nullptr;
    auto* tanh = TRT_ENGINE_ADD_LAYER(
        engine_, Activation, *input, nvinfer1::ActivationType::kTANH);
    layer = TRT_ENGINE_ADD_LAYER(engine_,
                                 ElementWise,
                                 *input,
                                 *(tanh->getOutput(0)),
                                 nvinfer1::ElementWiseOperation::kSUB);
    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "tanh_shrink", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(tanh_shrink, TanhshrinkOpConverter);
