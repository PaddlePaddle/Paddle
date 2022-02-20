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

// LeakyRelu converter from fluid to tensorRT
class LeakyReluOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert fluid leaky_relu op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    // Get attrs
    float alpha = BOOST_GET_CONST(float, op_desc.GetAttr("alpha"));
    nvinfer1::ILayer* output_layer = nullptr;

#if IS_TRT_VERSION_GE(5100)
    nvinfer1::IActivationLayer* layer = TRT_ENGINE_ADD_LAYER(
        engine_, Activation, *input, nvinfer1::ActivationType::kLEAKY_RELU);
    layer->setAlpha(alpha);
    output_layer = layer;

    bool enable_int8 = op_desc.HasAttr("enable_int8");
    if (enable_int8) {
      CHECK(op_desc.HasAttr("X_scale"));
      float in_scale = BOOST_GET_CONST(float, op_desc.GetAttr("X_scale"));
      engine_->SetTensorDynamicRange(input, in_scale);
    }
#else
    platform::CPUPlace place;
    std::unique_ptr<framework::LoDTensor> alpha_tensor(
        new framework::LoDTensor());
    alpha_tensor->Resize(phi::make_ddim({2}));
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
    PADDLE_ENFORCE_NOT_NULL(
        scale_layer, platform::errors::InvalidArgument(
                         "Invalid scale layer in leaky_relu TRT op converter. "
                         "The scale layer should not be null."));
    // y_relu = (x > 0) : x : 0
    auto* relu_layer = TRT_ENGINE_ADD_LAYER(engine_, Activation, *input,
                                            nvinfer1::ActivationType::kRELU);
    PADDLE_ENFORCE_NOT_NULL(
        relu_layer, platform::errors::InvalidArgument(
                        "Invalid relu layer in leaky_relu TRT op converter. "
                        "The relu layer should not be null."));
    //
    TensorRTEngine::Weight sub_scale{nvinfer1::DataType::kFLOAT, &alpha_data[1],
                                     1};
    auto* scale_relu_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Scale, *(relu_layer->getOutput(0)),
                             nvinfer1::ScaleMode::kUNIFORM, shift.get(),
                             sub_scale.get(), power.get());
    PADDLE_ENFORCE_NOT_NULL(
        scale_relu_layer,
        platform::errors::InvalidArgument(
            "Invalid scale_relu layer in leaky_relu TRT op converter. The "
            "scale_relu layer should not be null."));
    output_layer =
        TRT_ENGINE_ADD_LAYER(engine_, ElementWise, *(scale_layer->getOutput(0)),
                             *(scale_relu_layer->getOutput(0)),
                             nvinfer1::ElementWiseOperation::kSUM);
    PADDLE_ENFORCE_NOT_NULL(
        output_layer, platform::errors::InvalidArgument(
                          "Invalid output layer in leaky_relu TRT op "
                          "converter. The output layer should not be null."));
    // keep alpha tensor to avoid release it's memory
    std::string alpha_name = op_desc.Output("Out")[0] + "_alpha";
    bool alpha_not_in_weight_map =
        (engine_->weight_map.find(alpha_name) == engine_->weight_map.end());
    PADDLE_ENFORCE_EQ(alpha_not_in_weight_map, true,
                      platform::errors::InvalidArgument(
                          "The name of parameter alpha in leaky_relu TRT op "
                          "converter is already "
                          "found in the weight map. The same weight cannot be "
                          "set twice. Please check if it is already set."));
    engine_->SetWeights(alpha_name, std::move(alpha_tensor));
#endif
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(output_layer, "leaky_relu", {output_name},
                             test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(leaky_relu, LeakyReluOpConverter);
