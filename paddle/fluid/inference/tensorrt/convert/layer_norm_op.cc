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
#include "paddle/fluid/inference/tensorrt/plugin/layer_norm_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class LayerNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid batch norm op to tensorrt batch_norm";

    framework::OpDesc op_desc(op, nullptr);
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
    PADDLE_ENFORCE_EQ(op_desc.Input("Bias").size(), 1);   // Bias is a weight
    PADDLE_ENFORCE_EQ(op_desc.Input("Scale").size(), 1);  // Scale is a weight
    PADDLE_ENFORCE_EQ(op_desc.Output("Y").size(), 1);

    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    auto input_dims = X->getDimensions();
    // Declare weights
    auto* Bias_v = scope.FindVar(op_desc.Input("Bias").front());
    auto* Scale_v = scope.FindVar(op_desc.Input("Scale").front());

    PADDLE_ENFORCE_NOT_NULL(Bias_v);
    PADDLE_ENFORCE_NOT_NULL(Scale_v);

    float epsilon = boost::get<float>(op_desc.GetAttr("epsilon"));
    int begin_norm_axis = boost::get<int>(op_desc.GetAttr("begin_norm_axis"));

    PADDLE_ENFORCE(begin_norm_axis != 0);
    begin_norm_axis += (begin_norm_axis < 0) ? input_dims.nbDims : -1;

    // get tensor
    auto* Bias_t = Bias_v->GetMutable<framework::LoDTensor>();
    auto* Scale_t = Scale_v->GetMutable<framework::LoDTensor>();

    std::unique_ptr<framework::LoDTensor> scale_tensor(
        new framework::LoDTensor());
    std::unique_ptr<framework::LoDTensor> bias_tensor(
        new framework::LoDTensor());

    bias_tensor->Resize(Bias_t->dims());
    scale_tensor->Resize(Scale_t->dims());

    platform::CUDAPlace gpu_place;
    // copy data from gpu to gpu
    TensorCopySync((*Bias_t), gpu_place, bias_tensor.get());
    TensorCopySync((*Scale_t), gpu_place, scale_tensor.get());

    auto* bias_data = bias_tensor->mutable_data<float>(gpu_place);
    auto* scale_data = scale_tensor->mutable_data<float>(gpu_place);

    TensorRTEngine::Weight scale_w(nvinfer1::DataType::kFLOAT,
                                   static_cast<void*>(scale_data),
                                   scale_tensor->numel());
    TensorRTEngine::Weight bias_w(nvinfer1::DataType::kFLOAT,
                                  static_cast<void*>(bias_data),
                                  bias_tensor->numel());

    plugin::LayerNormPlugin* plugin =
        new plugin::LayerNormPlugin(scale_w, bias_w, epsilon, begin_norm_axis);
    nvinfer1::IPluginLayer* layer = engine_->AddPlugin(&X, 1, plugin);

    auto output_name = op_desc.Output("Y").front();
    layer->setName(("layer_norm (Output: " + output_name + ")").c_str());
    layer->getOutput(0)->setName(output_name.c_str());
    engine_->weight_map[op_desc.Input("Bias").front()] = std::move(bias_tensor);
    engine_->weight_map[op_desc.Input("Scale").front()] =
        std::move(scale_tensor);

    engine_->SetITensor(output_name, layer->getOutput(0));

    if (test_mode) {
      engine_->DeclareOutput(output_name);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(layer_norm, LayerNormOpConverter);
