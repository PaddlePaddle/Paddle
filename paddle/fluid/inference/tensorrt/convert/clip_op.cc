/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
 * ClipOp
 */
class ClipOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
#if IS_TRT_VERSION_GE(5130)
    VLOG(3) << "convert a clip op to tensorrt IActivationLayer.";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    float min = PADDLE_GET_CONST(float, op_desc.GetAttr("min"));
    float max = PADDLE_GET_CONST(float, op_desc.GetAttr("max"));
    nvinfer1::DataType input_type = input->getType();
    nvinfer1::DataType FP32_type = nvinfer1::DataType::kFLOAT;
    if (input_type != FP32_type) {
      input = Cast(input, FP32_type);
    }

    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, Activation, *input, nvinfer1::ActivationType::kCLIP);
    layer->setAlpha(min);
    layer->setBeta(max);
    nvinfer1::ILayer* out_layer = layer;
    auto output_name = op_desc.Output("Out")[0];
    if (input_type != FP32_type) {
      auto* temp_out = layer->getOutput(0);
      auto* layer_ = TRT_ENGINE_ADD_LAYER(engine_, Identity, *temp_out);
      layer_->setOutputType(0, input_type);
      layer_->getOutput(0)->setType(input_type);
      out_layer = layer_;
    }

    ReplenishLayerAndOutput(out_layer, "clip", {output_name}, test_mode);
#else
    PADDLE_THROW(
        platform::errors::Fatal("clip TRT converter is only supported on TRT "
                                "5.1.3.0 or higher version."));
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(clip, ClipOpConverter);
