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

#include <math.h>
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class BatchNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid batch norm op to tensorrt batch_norm";

    framework::OpDesc op_desc(op, nullptr);
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
    PADDLE_ENFORCE_EQ(op_desc.Input("Bias").size(), 1);   // Bias is a weight
    PADDLE_ENFORCE_EQ(op_desc.Input("Mean").size(), 1);   // Mean is a weight
    PADDLE_ENFORCE_EQ(op_desc.Input("Scale").size(), 1);  // Scale is a weight
    PADDLE_ENFORCE_EQ(op_desc.Input("Variance").size(),
                      1);  // Variance is a weight
    PADDLE_ENFORCE_EQ(op_desc.Output("Y").size(), 1);

    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    // Declare weights
    auto* Bias_v = scope.FindVar(op_desc.Input("Bias").front());
    auto* Mean_v = scope.FindVar(op_desc.Input("Mean").front());
    auto* Scale_v = scope.FindVar(op_desc.Input("Scale").front());
    auto* Variance_v = scope.FindVar(op_desc.Input("Variance").front());
    const float eps = boost::get<float>(op_desc.GetAttr("epsilon"));

    PADDLE_ENFORCE_NOT_NULL(Bias_v);
    PADDLE_ENFORCE_NOT_NULL(Mean_v);
    PADDLE_ENFORCE_NOT_NULL(Scale_v);
    PADDLE_ENFORCE_NOT_NULL(Variance_v);

    // get tensor
    auto* Bias_t = Bias_v->GetMutable<framework::LoDTensor>();
    auto* Mean_t = Mean_v->GetMutable<framework::LoDTensor>();
    auto* Scale_t = Scale_v->GetMutable<framework::LoDTensor>();
    auto* Variance_t = Variance_v->GetMutable<framework::LoDTensor>();

    // create temp tensor for weights
    framework::LoDTensor bias_tensor;
    framework::LoDTensor mean_tensor;
    framework::LoDTensor scale_tensor;
    framework::LoDTensor variance_tensor;

    bias_tensor.Resize(Bias_t->dims());
    mean_tensor.Resize(Mean_t->dims());
    scale_tensor.Resize(Scale_t->dims());
    variance_tensor.Resize(Variance_t->dims());

    platform::CPUPlace cpu_place;
    // copy data from gpu to cpu
    TensorCopySync((*Bias_t), cpu_place, &bias_tensor);
    TensorCopySync((*Mean_t), cpu_place, &mean_tensor);
    TensorCopySync((*Scale_t), cpu_place, &scale_tensor);
    TensorCopySync((*Variance_t), cpu_place, &variance_tensor);

    auto* bias_data = bias_tensor.mutable_data<float>(platform::CPUPlace());
    auto* mean_data = mean_tensor.mutable_data<float>(platform::CPUPlace());
    auto* scale_data = scale_tensor.mutable_data<float>(platform::CPUPlace());
    auto* variance_data =
        variance_tensor.mutable_data<float>(platform::CPUPlace());

    std::unique_ptr<framework::LoDTensor> combile_scale_tensor(
        new framework::LoDTensor());
    std::unique_ptr<framework::LoDTensor> combile_bias_tensor(
        new framework::LoDTensor());

    combile_scale_tensor->Resize(scale_tensor.dims());
    combile_bias_tensor->Resize(bias_tensor.dims());

    auto* combile_scale_data =
        combile_scale_tensor->mutable_data<float>(platform::CPUPlace());
    auto* combile_bias_data =
        combile_bias_tensor->mutable_data<float>(platform::CPUPlace());

    size_t ele_num = combile_scale_tensor->memory_size() / sizeof(float);

    for (size_t i = 0; i < ele_num; i++) {
      float scale = scale_data[i];
      float bias = bias_data[i];
      float mean = mean_data[i];
      float variance = variance_data[i];
      combile_scale_data[i] = scale / sqrtf(variance + eps);
      combile_bias_data[i] = bias - mean * combile_scale_data[i];
    }

    TensorRTEngine::Weight scale_weights{
        nvinfer1::DataType::kFLOAT, static_cast<void*>(combile_scale_data),
        combile_scale_tensor->memory_size() / sizeof(float)};
    TensorRTEngine::Weight shift_weights{
        nvinfer1::DataType::kFLOAT, static_cast<void*>(combile_bias_data),
        combile_bias_tensor->memory_size() / sizeof(float)};
    TensorRTEngine::Weight power_weights{nvinfer1::DataType::kFLOAT, nullptr,
                                         0};

    nvinfer1::IScaleLayer* layer =
        TRT_ENGINE_ADD_LAYER(engine_, Scale, *const_cast<nvinfer1::ITensor*>(X),
                             nvinfer1::ScaleMode::kCHANNEL, shift_weights.get(),
                             scale_weights.get(), power_weights.get());

    auto output_name = op_desc.Output("Y").front();
    engine_->weight_map[op_desc.Input("Bias").front()] =
        std::move(combile_bias_tensor);
    engine_->weight_map[op_desc.Input("Scale").front()] =
        std::move(combile_scale_tensor);

    RreplenishLayerAndOutput(layer, "pool2d", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(batch_norm, BatchNormOpConverter);
