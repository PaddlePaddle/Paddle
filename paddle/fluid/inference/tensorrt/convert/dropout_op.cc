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

/*
 * DropoutOp. This Layer doesn't has weights.
 */
class DropoutOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid dropout op to tensorrt dropout layer";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input1 = engine_->GetITensor(op_desc.Input("X")[0]);
    float dropout_prob =
        BOOST_GET_CONST(float, op_desc.GetAttr("dropout_prob"));

    std::string downgrade_in_infer = "";
    if (op_desc.HasAttr("dropout_implementation")) {
      downgrade_in_infer = BOOST_GET_CONST(
          std::string, op_desc.GetAttr("dropout_implementation"));
    }

    if (!downgrade_in_infer.empty() &&
        downgrade_in_infer == "upscale_in_train") {
      auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input1);
      auto output_name = op_desc.Output("Out")[0];
      RreplenishLayerAndOutput(layer, "dropout", {output_name}, test_mode);
      return;
    }

    platform::CPUPlace cpu_place;
    std::unique_ptr<framework::LoDTensor> weight_tensor(
        new framework::LoDTensor());
    weight_tensor->Resize(phi::make_ddim({1}));
    auto* weight_data =
        weight_tensor->mutable_data<float>(platform::CPUPlace());
    weight_data[0] = 1 - dropout_prob;

    TensorRTEngine::Weight scale_weights{
        nvinfer1::DataType::kFLOAT, static_cast<void*>(weight_data),
        weight_tensor->memory_size() / sizeof(float)};
    TensorRTEngine::Weight shift_weights{nvinfer1::DataType::kFLOAT, nullptr,
                                         0};
    TensorRTEngine::Weight power_weights{nvinfer1::DataType::kFLOAT, nullptr,
                                         0};

    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, Scale, *const_cast<nvinfer1::ITensor*>(input1),
        nvinfer1::ScaleMode::kUNIFORM, shift_weights.get(), scale_weights.get(),
        power_weights.get());

    engine_->SetWeights(op_desc.Output("Out").front() + "_dropout",
                        std::move(weight_tensor));
    auto output_name = op_desc.Output("Out")[0];

    RreplenishLayerAndOutput(layer, "dropout", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP_ITSELF(dropout);
REGISTER_TRT_OP_CONVERTER(dropout, DropoutOpConverter);
