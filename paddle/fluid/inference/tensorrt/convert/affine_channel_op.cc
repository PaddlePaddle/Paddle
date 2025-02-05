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

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle::inference::tensorrt {

/*
 * Affine Channel Op
 */
class AffineChannelOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a affine_channel op to tensorrt scale nd layer";

    framework::OpDesc op_desc(op, nullptr);
    std::string input_name = op_desc.Input("X").front();
    std::string scale_name = op_desc.Input("Scale").front();
    std::string bias_name = op_desc.Input("Bias").front();
    std::string output_name = op_desc.Output("Out").front();

    auto input_tensor = engine_->GetITensor(input_name);
    auto input_dim = input_tensor->getDimensions();

    auto* scale_v = scope.FindVar(scale_name);
    auto* scale_t = scale_v->GetMutable<phi::DenseTensor>();
    float* scale_ptr = const_cast<float*>(static_cast<const float*>(
        engine_->GetFp32TrtWeight(scale_name, *scale_t).get().values));

    auto* bias_v = scope.FindVar(bias_name);
    auto* bias_t = bias_v->GetMutable<phi::DenseTensor>();
    float* bias_ptr = const_cast<float*>(static_cast<const float*>(
        engine_->GetFp32TrtWeight(bias_name, *bias_t).get().values));

    // tensorrt scalend layer only support spatial dims >= 2,
    // so nhwc is not available (spatial dims == 0)
    const int channel_axis = 1;

    TensorRTEngine::Weight scale_weights{
        nvinfer1::DataType::kFLOAT,
        static_cast<void*>(scale_ptr),
        static_cast<size_t>(input_dim.d[channel_axis])};
    TensorRTEngine::Weight bias_weights{
        nvinfer1::DataType::kFLOAT,
        static_cast<void*>(bias_ptr),
        static_cast<size_t>(input_dim.d[channel_axis])};
    TensorRTEngine::Weight power_weights{
        nvinfer1::DataType::kFLOAT, nullptr, 0};

    auto layer = TRT_ENGINE_ADD_LAYER(engine_,
                                      ScaleNd,
                                      *input_tensor,
                                      nvinfer1::ScaleMode::kCHANNEL,
                                      bias_weights.get(),
                                      scale_weights.get(),
                                      power_weights.get(),
                                      channel_axis);

    ReplenishLayerAndOutput(layer, "affine_channel", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(affine_channel, AffineChannelOpConverter);
