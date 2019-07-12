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
namespace inference {
namespace tensorrt {

// LeakyRelu converter from fluid to tensorRT
class LeakyReluOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert fluid leaky_relu op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    int input_num = op_desc.Input("X").size();
    PADDLE_ENFORCE(input_num == 1);
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    // Get output
    size_t output_num = op_desc.Output("Out").size();
    PADDLE_ENFORCE(output_num == 1);
    // Get attrs
    float alpha = boost::get<float>(op_desc.GetAttr("alpha"));

    platform::CPUPlace place;
    std::unique_ptr<framework::LoDTensor> alpha_tensor(
        new framework::LoDTensor());
    alpha_tensor->Resize(framework::make_ddim({2}));
    float* alpha_data = alpha_tensor->mutable_data<float>(place);
    alpha_data[0] = alpha;
    alpha_data[1] = 1.f - alpha;
    // the leaky relu formula y = (x > 0) ? x : alpha * x is equal to
    // y = alpha * x + (x > 0) ? (1 - alpha) * x : 0
    TensorRTEngine::Weight scale{nvinfer1::DataType::kFLOAT, &alpha_data[0], 1};
    TensorRTEngine::Weight shift{nvinfer1::DataType::kFLOAT, nullptr, 0};
    TensorRTEngine::Weight power{nvinfer1::DataType::kFLOAT, nullptr, 0};
    // y_scale = alpha * x
    auto* scale_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Scale, *input, nvinfer1::ScaleMode::kUNIFORM, shift.get(),
        scale.get(), power.get());
    PADDLE_ENFORCE(nullptr != scale_layer);
    // y_relu = (x > 0) : x : 0
    auto* relu_layer = TRT_ENGINE_ADD_LAYER(engine_, Activation, *input,
                                            nvinfer1::ActivationType::kRELU);
    PADDLE_ENFORCE(nullptr != relu_layer);
    //
    TensorRTEngine::Weight sub_scale{nvinfer1::DataType::kFLOAT, &alpha_data[1],
                                     1};
    auto* scale_relu_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Scale, *(relu_layer->getOutput(0)),
                             nvinfer1::ScaleMode::kUNIFORM, shift.get(),
                             sub_scale.get(), power.get());
    PADDLE_ENFORCE(nullptr != scale_relu_layer);
    auto* output_layer =
        TRT_ENGINE_ADD_LAYER(engine_, ElementWise, *(scale_layer->getOutput(0)),
                             *(scale_relu_layer->getOutput(0)),
                             nvinfer1::ElementWiseOperation::kSUM);
    PADDLE_ENFORCE(nullptr != output_layer);
    // keep alpha tensor to avoid release it's memory
    std::string alpha_name = op_desc.Output("Out")[0] + "_alpha";
    PADDLE_ENFORCE(engine_->weight_map.find(alpha_name) ==
                   engine_->weight_map.end());
    engine_->weight_map[alpha_name] = std::move(alpha_tensor);

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(output_layer, "leaky_relu", {output_name},
                             test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(leaky_relu, LeakyReluOpConverter);
