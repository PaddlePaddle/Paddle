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
#include "paddle/fluid/inference/tensorrt/plugin/layer_norm_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class LayerNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert a fluid layer_norm op to tensorrt layer_norm plugin";

    framework::OpDesc op_desc(op, nullptr);
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
    PADDLE_ENFORCE_EQ(op_desc.Input("Bias").size(), 1);   // Bias is a weight
    PADDLE_ENFORCE_EQ(op_desc.Input("Scale").size(), 1);  // Scale is a weight

    PADDLE_ENFORCE_EQ(op_desc.Output("Y").size(), 1);

    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    // Declare weights
    auto* Bias_v = scope.FindVar(op_desc.Input("Bias").front());
    auto* Scale_v = scope.FindVar(op_desc.Input("Scale").front());
    const float eps = boost::get<float>(op_desc.GetAttr("epsilon"));

    PADDLE_ENFORCE_NOT_NULL(Bias_v);
    PADDLE_ENFORCE_NOT_NULL(Scale_v);

    // get tensor
    auto* Bias_t = Bias_v->GetMutable<framework::LoDTensor>();
    auto* Scale_t = Scale_v->GetMutable<framework::LoDTensor>();

    // create temp tensor for weights
    framework::LoDTensor bias_tensor;
    framework::LoDTensor scale_tensor;

    bias_tensor.Resize(Bias_t->dims());
    scale_tensor.Resize(Scale_t->dims());

    platform::CPUPlace cpu_place;
    // copy data from gpu to cpu
    TensorCopySync((*Bias_t), cpu_place, &bias_tensor);
    TensorCopySync((*Scale_t), cpu_place, &scale_tensor);

    auto* bias_data = bias_tensor.mutable_data<float>(platform::CPUPlace());
    auto* scale_data = scale_tensor.mutable_data<float>(platform::CPUPlace());

    std::unique_ptr<framework::LoDTensor> combine_scale_tensor(
        new framework::LoDTensor());
    std::unique_ptr<framework::LoDTensor> combine_bias_tensor(
        new framework::LoDTensor());

    combine_scale_tensor->Resize(scale_tensor.dims());
    combine_bias_tensor->Resize(bias_tensor.dims());

    auto* combine_scale_data =
        combine_scale_tensor->mutable_data<float>(platform::CPUPlace());
    auto* combine_bias_data =
        combine_bias_tensor->mutable_data<float>(platform::CPUPlace());

    size_t ele_num = combine_scale_tensor->memory_size() / sizeof(float);

    for (size_t i = 0; i < ele_num; i++) {
      float scale = scale_data[i];
      float bias = bias_data[i];
      float mean = mean_data[i];
      float variance = variance_data[i];
      combine_scale_data[i] = scale / sqrtf(variance + eps);
      combine_bias_data[i] = bias - mean * combine_scale_data[i];
    }

    TensorRTEngine::Weight scale_weights{
        nvinfer1::DataType::kFLOAT, static_cast<void*>(combine_scale_data),
        combine_scale_tensor->memory_size() / sizeof(float)};
    TensorRTEngine::Weight shift_weights{
        nvinfer1::DataType::kFLOAT, static_cast<void*>(combine_bias_data),
        combine_bias_tensor->memory_size() / sizeof(float)};
    TensorRTEngine::Weight power_weights{nvinfer1::DataType::kFLOAT, nullptr,
                                         0};

    nvinfer1::IScaleLayer* layer =
        TRT_ENGINE_ADD_LAYER(engine_, Scale, *const_cast<nvinfer1::ITensor*>(X),
                             nvinfer1::ScaleMode::kCHANNEL, shift_weights.get(),
                             scale_weights.get(), power_weights.get());

    auto output_name = op_desc.Output("Y").front();
    engine_->SetWeights(op_desc.Input("Bias").front(),
                        std::move(combine_bias_tensor));
    engine_->SetWeights(op_desc.Input("Scale").front(),
                        std::move(combine_scale_tensor));
    RreplenishLayerAndOutput(layer, "batch_norm", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP(layer_norm);
REGISTER_TRT_OP_CONVERTER(layer_norm, LayerNormOpConverter);
