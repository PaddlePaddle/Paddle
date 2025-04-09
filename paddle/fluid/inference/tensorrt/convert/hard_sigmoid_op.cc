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

namespace paddle::inference::tensorrt {

/*
 * HardSigmoidOp, IActivationLayer in TRT. This Layer doesn't has weights.
 */
class HardSigmoidOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
#if IS_TRT_VERSION_GE(5130)
    VLOG(3) << "convert a HardSigmoid op to tensorrt IActivationLayer "
               "layer without bias";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    float slope = PADDLE_GET_CONST(float, op_desc.GetAttr("slope"));
    float offset = PADDLE_GET_CONST(float, op_desc.GetAttr("offset"));
    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, Activation, *input, nvinfer1::ActivationType::kHARD_SIGMOID);
    layer->setAlpha(slope);
    layer->setBeta(offset);

    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "hard_sigmoid", {output_name}, test_mode);
#else
    PADDLE_THROW(common::errors::Fatal(
        "Hard sigmoid TRT converter is only supported on TRT 5 or higher. "
        "Please confirm your TRT version is no less than 5.0."));
#endif
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(hard_sigmoid, HardSigmoidOpConverter);
