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

/*
 * ConcatOp
 */
class ScaleOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid scale op to tensorrt mul layer without bias";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    std::vector<nvinfer1::ITensor*> itensors;
    std::string input_name = op_desc.Input("X").front();
    std::string out_name = op_desc.Output("Out").front();

    auto input = engine_->GetITensor(input_name);
    bool bias_after_scale =
        BOOST_GET_CONST(bool, op_desc.GetAttr("bias_after_scale"));
    float bias = BOOST_GET_CONST(float, op_desc.GetAttr("bias"));
    float scale = BOOST_GET_CONST(float, op_desc.GetAttr("scale"));
    auto create_weights = [&](float data, std::string type) -> float* {
      std::unique_ptr<framework::Tensor> tmp_tensor(new framework::Tensor());
      tmp_tensor->Resize({1});
      auto* tmp_data = tmp_tensor->mutable_data<float>(platform::CPUPlace());
      tmp_data[0] = data;
      engine_->SetWeights(out_name + "_scale_op_" + type,
                          std::move(tmp_tensor));
      return tmp_data;
    };

    float* bias_ptr = create_weights(bias, "bias");
    float* scale_ptr = create_weights(scale, "scale");

    TensorRTEngine::Weight scale_weights{nvinfer1::DataType::kFLOAT,
                                         static_cast<void*>(scale_ptr), 1};
    TensorRTEngine::Weight shift_weights{nvinfer1::DataType::kFLOAT,
                                         static_cast<void*>(bias_ptr), 1};
    TensorRTEngine::Weight power_weights{nvinfer1::DataType::kFLOAT, nullptr,
                                         0};
    nvinfer1::ILayer* layer = nullptr;
    if (bias_after_scale) {
      layer = TRT_ENGINE_ADD_LAYER(
          engine_, Scale, *input, nvinfer1::ScaleMode::kUNIFORM,
          shift_weights.get(), scale_weights.get(), power_weights.get());
    } else {
      // add bias
      layer = TRT_ENGINE_ADD_LAYER(
          engine_, Scale, *(input), nvinfer1::ScaleMode::kUNIFORM,
          shift_weights.get(), power_weights.get(), power_weights.get());
      // mul scale
      layer = TRT_ENGINE_ADD_LAYER(
          engine_, Scale, *(layer->getOutput(0)), nvinfer1::ScaleMode::kUNIFORM,
          power_weights.get(), scale_weights.get(), power_weights.get());
    }

    RreplenishLayerAndOutput(layer, "scale", {out_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(scale, ScaleOpConverter);
